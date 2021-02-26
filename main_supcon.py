from __future__ import print_function

import os
import sys
import argparse
import time
import math

import tensorboard_logger as tb_logger
import torch
import popart
import poptorch
from torchvision import transforms, datasets

from util import TwoCropTransform, AverageMeter
from util import adjust_learning_rate, warmup_learning_rate
from util import pipeline_model, replace_bn, set_optimizer, save_model
from networks.resnet_big import SupConResNet
from losses import SupConLoss


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='number of training epochs')

    # IPU options
    parser.add_argument('--pipeline_splits', nargs='+', help='pipeline splits')
    parser.add_argument('--enable_pipeline_recompute', action='store_true',
                        help='Enable the recomputation of network activations during backward pass instead of caching them during forward pass')
    parser.add_argument('--gradient_accumulation', type=int, default=1,
                        help='gradient accumulation')
    parser.add_argument('--replication_factor', type=int, default=1,
                        help='replication factor')
    parser.add_argument('--memory_proportion', type=float, default=0.6,
                        help='available memory proportion for conv and matmul')
    parser.add_argument('--norm_type', default='batch', choices=['batch', 'group', 'none'],
                        help='normalization layer type')
    parser.add_argument('--norm_num_group', type=int, default=32,
                        help='number of groups for group normalization layers')
    parser.add_argument('--precision', default='16.16', choices=['16.16', '16.32', '32.32'],
                        help='Precision of Ops(weights/activations/gradients) and Master data types: 16.16, 16.32, 32.32')
    parser.add_argument('--half_partial', action='store_true',
                        help='Accumulate matrix multiplication partials in half precision')

    # optimization
    parser.add_argument('--optimizer', default='SGD', choices=['SGD', 'Adam', 'RMSprop'],
                        help='optimizer for training')
    parser.add_argument('--learning_rate', type=float, default=0.05,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='700,800,900',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    parser.add_argument('--betas', type=float, nargs=2, default=[0.9, 0.999],
                        help='betas for Adam optimizer')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'path'], help='dataset')
    parser.add_argument('--mean', type=str, help='mean of dataset in path in form of str tuple')
    parser.add_argument('--std', type=str, help='std of dataset in path in form of str tuple')
    parser.add_argument('--data_folder', type=str, default=None, help='path to custom dataset')
    parser.add_argument('--size', type=int, default=32, help='parameter for RandomResizedCrop')

    # method
    parser.add_argument('--method', type=str, default='SupCon',
                        choices=['SupCon', 'SimCLR'], help='choose method')

    # temperature
    parser.add_argument('--temp', type=float, default=0.07,
                        help='temperature for loss function')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')

    opt = parser.parse_args()

    # check if dataset is path that passed required arguments
    if opt.dataset == 'path':
        assert opt.data_folder is not None \
            and opt.mean is not None \
            and opt.std is not None

    # set the path according to the environment
    if opt.data_folder is None:
        opt.data_folder = './datasets/'
    opt.model_path = './save/SupCon/{}_models'.format(opt.dataset)
    opt.tb_path = './save/SupCon/{}_tensorboard'.format(opt.dataset)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_{}_lr_{}_decay_{}_bsz_{}_temp_{}_trial_{}'.\
        format(opt.method, opt.dataset, opt.model, opt.learning_rate,
               opt.weight_decay, opt.batch_size, opt.temp, opt.trial)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
    if opt.batch_size > 256:
        opt.warm = True
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    opt.profiling = 'POPLAR_ENGINE_OPTIONS' in os.environ
    assert len(opt.pipeline_splits) in (0, 1, 3, 7, 15)

    return opt


def set_loader(opt, poptorch_opts: poptorch.Options):
    # construct data loader
    if opt.dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif opt.dataset == 'cifar100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    elif opt.dataset == 'path':
        mean = eval(opt.mean)
        std = eval(opt.mean)
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))
    normalize = transforms.Normalize(mean=mean, std=std)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=opt.size, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        normalize,
    ])

    if opt.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root=opt.data_folder,
                                         transform=TwoCropTransform(train_transform),
                                         download=True)
    elif opt.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(root=opt.data_folder,
                                          transform=TwoCropTransform(train_transform),
                                          download=True)
    elif opt.dataset == 'path':
        train_dataset = datasets.ImageFolder(root=opt.data_folder,
                                            transform=TwoCropTransform(train_transform))
    else:
        raise ValueError(opt.dataset)

    train_sampler = None
    train_loader = poptorch.DataLoader(
        poptorch_opts, train_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
        num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)

    train_loader.__len__ = lambda: len(train_dataset)

    return train_loader


class ModelWithLoss(torch.nn.Module):
    def __init__(self, model, loss, method):
        super().__init__()
        self.model = model
        self.loss = loss
        self.method = method

    def forward(self, x, labels):
        bsz = labels.shape[0]
        features = self.model(x)
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

        if self.method == 'SupCon':
            loss = self.loss(features, labels)
        elif self.method == 'SimCLR':
            loss = self.loss(features)
        else:
            raise ValueError('contrastive method not supported: {}'.
                             format(self.method))

        return features, loss


def set_model(opt):
    model = SupConResNet(name=opt.model)
    criterion = SupConLoss(temperature=opt.temp)

    if opt.norm_type in ['group', 'none']:
        replace_bn(model, 'group', opt.norm_num_group)

    pipeline_model(model, opt.pipeline_splits)

    model_with_loss = ModelWithLoss(model, criterion, opt.method).train()

    if opt.precision[-3:] == ".16":
        model.half()

    return model_with_loss


def train(train_loader, model, optimizer, epoch, opt):
    """one epoch training"""

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):
        if opt.profiling and idx > 0:
            break
        data_time.update(time.time() - end)

        images = torch.cat([images[0], images[1]], dim=0)
        if opt.precision[:2] == "16":
            images = images.half()
        bsz = labels.shape[0]
        labels = labels.int()

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)
        model.setOptimizer(optimizer)

        # compute loss
        features, loss = model(images, labels)

        # update metric
        losses.update(loss.item(), bsz)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))
            sys.stdout.flush()

    return losses.avg


def main():
    opt = parse_option()

    # poptorch options
    poptorch_opts = poptorch.Options()
    poptorch_opts.Training.gradientAccumulation(opt.gradient_accumulation)
    poptorch_opts.replicationFactor(opt.replication_factor)
    poptorch_opts.Training.accumulationReductionType(poptorch.ReductionType.Mean)
    poptorch_opts.setAvailableMemoryProportion({
        f'IPU{ipu_id}': opt.memory_proportion for ipu_id in range(len(opt.pipeline_splits))
    })
    if opt.half_partial:
        poptorch_opts.Popart.set("partialsTypeMatMuls", "half")
        poptorch_opts.Popart.set("convolutionOptions", {'partialsType': 'half'})
    if opt.enable_pipeline_recompute and len(opt.pipeline_splits) > 0:
        poptorch_opts.Popart.set("autoRecomputation", int(popart.RecomputationType.Pipeline))

    # build data loader
    train_loader = set_loader(opt, poptorch_opts)

    # build model and criterion
    model = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, model)

    # poptorch wrapper
    poptorch_model = poptorch.trainingModel(model, options=poptorch_opts, optimizer=optimizer)

    # tensorboard
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    # training routine
    for epoch in range(1, opt.epochs + 1):
        if opt.profiling and epoch > 1:
            break
        adjust_learning_rate(opt, optimizer, epoch)
        poptorch_model.setOptimizer(optimizer)

        # train for one epoch
        time1 = time.time()
        loss = train(train_loader, poptorch_model, optimizer, epoch, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        # tensorboard logger
        logger.log_value('loss', loss, epoch)
        logger.log_value('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        if epoch % opt.save_freq == 0:
            save_file = os.path.join(
                opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            poptorch_model.copyWeightsToHost()
            save_model(model, optimizer, opt, epoch, save_file)

    # save the last model
    save_file = os.path.join(
        opt.save_folder, 'last.pth')
    poptorch_model.copyWeightsToHost()
    save_model(model, optimizer, opt, opt.epochs, save_file)


if __name__ == '__main__':
    main()
