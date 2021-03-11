from __future__ import print_function

import math
import numpy as np
import torch
import poptorch
import poptorch.optim as optim


def replace_layer(parent, field_name, new_layer):
    if isinstance(parent, torch.nn.Sequential):
        parent[int(field_name)] = new_layer
    else:
        setattr(parent, field_name, new_layer)


def replace_bn(model, norm_type, norm_num_groups=32):
    stack = [model]
    while len(stack) != 0:
        node = stack.pop()
        for name, child in node.named_children():
            stack.append(child)
            if isinstance(child, torch.nn.BatchNorm2d):
                if norm_type == "group":
                    new_layer = torch.nn.GroupNorm(norm_num_groups, child.num_features, child.eps, child.affine)
                else:
                    new_layer = torch.nn.Identity()
                replace_layer(node, name, new_layer)


def get_module_and_parent_by_name(node, split_tokens):
    child_to_find = split_tokens[0]
    for name, child in node.named_children():
        if name == child_to_find:
            if len(split_tokens) == 1:
                return node, child, name
            else:
                return get_module_and_parent_by_name(child, split_tokens[1:])

    return None, None, None


def pipeline_model(model, pipeline_splits):
    """
    Split the model into stages.
    """
    for name, modules in model.named_modules():
        name = name.replace('.', '/')
        if name == 'encoder/conv1':
            parent, node, field_or_idx_str = get_module_and_parent_by_name(model, name.split('/'))
            if parent is None:
                raise Exception(f'Split {name} not found')
            else:
                replace_layer(parent, field_or_idx_str, poptorch.BeginBlock(ipu_id=0, layer_to_call=node))
        if name in pipeline_splits:
            print('--------')
        print(name)

    for split_idx, split in enumerate(pipeline_splits):
        split_tokens = split.split('/')
        print(f'Processing pipeline split {split_tokens}')
        parent, node, field_or_idx_str = get_module_and_parent_by_name(model, split_tokens)
        if parent is None:
            raise Exception(f'Split {split} not found')
        else:
            replace_layer(parent, field_or_idx_str, poptorch.BeginBlock(ipu_id=split_idx+1, layer_to_call=node))


class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def adjust_learning_rate(args, optimizer, epoch):
    lr = args.learning_rate
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def set_optimizer(opt, model):
    if opt.optimizer == 'SGD':
        return optim.SGD(model.parameters(),
                         lr=opt.learning_rate,
                         momentum=opt.momentum,
                         loss_scaling=opt.loss_scaling,
                        weight_decay=opt.weight_decay)
    elif opt.optimizer == 'Adam':
        return optim.Adam(model.parameters(),
                          lr=opt.learning_rate,
                          betas=opt.betas,
                          loss_scaling=opt.loss_scaling,
                          weight_decay=opt.weight_decay)
    elif opt.optimizer == 'RMSprop':
        return optim.RMSprop(model.parameters(),
                             lr=opt.learning_rate,
                             momentum=opt.momentum,
                             loss_scaling=opt.loss_scaling,
                             weight_decay=opt.weight_decay)
    else:
        raise Exception(f'Unknown optimizer: {opt.optimizer}')


def save_model(model, optimizer, opt, epoch, save_file):
    print('==> Saving...')
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state
