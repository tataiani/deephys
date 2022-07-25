'''
Testing script for ImageNet
Copyright (c) Wei YANG, 2017
'''
from __future__ import print_function

import argparse
import os
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import shutil
import time
import random
from enum import Enum
import pickle
import numpy as np
from collections import defaultdict
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
# import models.imagenet as customized_models
from transformers import AutoFeatureExtractor, ResNetForImageClassification

from SENN.utils import concept_grid

# from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig


# Parse arguments
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# Datasets
parser.add_argument('-d', '--data', default='/om5/user/xboix/data/ImageNet/raw-data', type=str)
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--test-batch', default=200, type=int, metavar='N',
                    help='test batchsize (default: 200)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
# parser.add_argument('--gpu-id', default='0', type=str,
#                     help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--gpu', default='0', type=str,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
args = parser.parse_args()
# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
use_cuda = torch.cuda.is_available()

args.distributed = args.world_size > 1 or args.multiprocessing_distributed

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
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

    def all_reduce(self):
        total = torch.FloatTensor([self.sum, self.count])
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)

        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


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
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def validate(val_loader, model, criterion, args):
    all_activs = []
    all_outputs = []
    activation = {}

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()

        return hook

    def run_validate(loader, base_progress=0):
        with torch.no_grad():
            end = time.time()
            for i, (images, target) in enumerate(loader):
                i = base_progress + i
                if args.gpu is not None:
                    images = images.cuda(int(args.gpu), non_blocking=True)
                if torch.cuda.is_available():
                    target = target.cuda(int(args.gpu), non_blocking=True)

                # compute output
                output = model(images).logits
                loss = criterion(output, target)

                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % args.print_freq == 0:
                    progress.display(i + 1)

                model.resnet.pooler.register_forward_hook(get_activation('linear1'))
                with torch.no_grad():
                    activation['output'] = model(images).logits
                    all_outputs.append(activation['output'])
                    activation['linear1'] = torch.squeeze(activation['linear1'])
                    all_activs.append(activation['linear1'])

    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_loader) + (args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset))),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    run_validate(val_loader)
    if args.distributed:
        top1.all_reduce()
        top5.all_reduce()

    if args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset)):
        aux_val_dataset = Subset(val_loader.dataset,
                                 range(len(val_loader.sampler) * args.world_size, len(val_loader.dataset)))
        aux_val_loader = torch.utils.data.DataLoader(
            aux_val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
        run_validate(aux_val_loader, len(val_loader))

    progress.display_summary()

    all_activs = torch.cat(all_activs)
    all_outputs = torch.cat(all_outputs)

    return top1.avg, all_activs, all_outputs

def concept_grid(model, data_loader, top_k = 1, layout = 'vertical', return_fig=False, save_path = None):
    """
        Finds examples in data_loader that are most representatives of concepts.

        For scalar concepts, activation is simply the value of the concept.
        For vector concepts, activation is the norm of the concept.

    """
    print('Warning: make sure data_loader passed to this function doesnt shuffle data!!')
    # all_norms = []
    num_concepts = 512 #model.conceptizer.nconcept
    # concept_dim  = model.conceptizer.dout

    # for i in range(num_concepts):
    #   path = 'test/imagenet_{}'.format(i+1)
    #   if not os.path.exists(path):
    #       os.mkdir(path)

    # def get_activation(name):
    #     def hook(model, input, output):
    #         activation[name] = output.detach()
    #
    #     return hook
    #
    # top_activations = {k: np.array(top_k*[-1000.00]) for k in range(num_concepts)}
    # top_examples = {k: top_k*[None] for k in range(num_concepts)}
    # all_activs = []
    # all_outputs = []
    # activation = {}
    # # for idx, (data, target) in enumerate(data_loader):
    # for i, (images, target) in enumerate(data_loader):
    #     # get the inputs
    #     # target = Variable(indexes.long())
    #     # if cuda:
    #     #     data, target = data.cuda(), target.cuda()
    #     # data, target = Variable(data, volatile=True), Variable(target)
    #     if args.gpu is not None:
    #         images = images.cuda(int(args.gpu), non_blocking=True)
    #     if torch.cuda.is_available():
    #         target = target.cuda(int(args.gpu), non_blocking=True)
    #
    #     # compute output
    #     output = model(images).logits
    #     # output = model(data)
    #     # concepts = model.concepts.data
    #     #pdb.set_trace()
    #     #concepts[concepts < 0] = 0.0 # This is unncessary if output of H is designed to be > 0.
    #     # if concepts.shape[-1] > 1:
    #     #     print('ERROR')
    #     #     print(asd.asd)
    #     #     activations = np.linalg.norm(concepts, axis = 2)
    #     # else:
    #     #     activations = concepts
    #
    #     model.resnet.pooler.register_forward_hook(get_activation('linear1'))
    #     with torch.no_grad():
    #         activation['output'] = model(images).logits
    #         all_outputs.append(activation['output'])
    #         activation['linear1'] = torch.squeeze(activation['linear1'])
    #         all_activs.append(activation['linear1'])
    #
    #     # all_activs.append(activations)
    #     # if idx == 10:
    #     #     break
    #
    # all_activs = torch.cat(all_activs)
    with open('imagetnet_activs.pkl', 'rb') as f:
        all_activs = pickle.load(f)
    top_activations, top_idxs = torch.topk(all_activs, top_k, 0)
    top_activations = top_activations.squeeze().t()
    top_idxs = top_idxs.squeeze().t()
    # top_examples = {}
    top_examples = defaultdict(list)
    # print (top_idxs)
    for i in range(num_concepts):
        # top_examples[i] = data_loader.dataset.test_data[top_idxs[i]]
        # top_examples[i] = data_loader.dataset[top_idxs[i]]
        for j in top_idxs[i]:
            get_image,_ = data_loader.dataset.samples[j]
            # get_image = mpimg.imread(get_image)
            # get_image_copy = get_image.copy()
            # get_image_copy = get_image_copy.resize(224, 224)
            img = Image.open(get_image)
            rsize = img.resize((224, 224))
            rsizeArr = np.asarray(rsize)
            # print (rsizeArr.shape)
            # print (rsizeArr)
            top_examples[i].append(rsizeArr)#.permute(1, 2, 0).numpy())
    #top_examples =
    print("Step 3")


    # Before, i was doing this manually :
        # for i in range(activations.shape[0]):
        #     #pdb.set_trace()
        #     for j in range(num_concepts):
        #         min_val  = top_activations[j].min()
        #         min_idx  = top_activations[j].argmin()
        #         if activations[i,j] >  min_val:
        #             # Put new one in place of min
        #             top_activations[j][min_idx]  = activations[i,j]
        #             top_examples[j][min_idx] = data[i, :, :, :].data.numpy().squeeze()
        #     #pdb.set_trace()
    # for k in range(num_concepts):
    #     #print(k)
    #     Z = [(v,e) for v,e in sorted(zip(top_activations[k],top_examples[k]),  key=lambda x: x[0], reverse = True)]
    #     top_activations[k], top_examples[k] = zip(*Z)

    if layout == 'horizontal':
        num_cols = top_k
        num_rows = num_concepts/2
        figsize=(num_cols, 1.2*num_rows)
    else:
        num_cols = 112#num_concepts/2
        num_rows = top_k
        figsize=(1.4*num_cols, num_rows)

    fig, axes  = plt.subplots(figsize=figsize, nrows=num_rows, ncols=num_cols )

    for i in range(112):
        for j in range(top_k):
            # print(top_examples[i][j].shape)
            # print (top_examples[i][j])
            # plt.imsave('/om2/user/anirbans/ImageNet/test/imagenet_'+ str(i+1)+'/'+str(j)+'.png', top_examples[i][j])
            # img = Image.fromarray(np.uint8(top_examples[i][j]), 'RGB')
            # img.save('/test/imagenet_'+ str(i+1)+'/'+str(j)+'.png')
            pos = (i,j) if layout == 'horizontal' else (j,i)

            l = i*top_k + j
            print(i,j)
            # print(top_examples[i][j].shape)
            # print (top_examples[i][j])
            # plt.imsave('/home/cs16resch11006/AwA/scripts/test/imagenet_'+ str(i+1)+'/'+str(j)+'.png', top_examples[i][j])
            axes[pos].imshow(top_examples[400+i][j], cmap='Greys',  interpolation='nearest')
            if layout == 'vertical':
                axes[pos].axis('off')
                if j == 0:
                    axes[pos].set_title('Cpt {}'.format(i+1), fontsize = 24)
            else:
                axes[pos].set_xticklabels([])
                axes[pos].set_yticklabels([])
                axes[pos].set_yticks([])
                axes[pos].set_xticks([])
                for side in ['top', 'right', 'bottom', 'left']:
                    axes[i,j].spines[side].set_visible(False)
                if i == 0:
                    axes[pos].set_title('Proto {}'.format(j+1))
                if j == 0:
                    axes[pos].set_ylabel('Concept {}'.format(i+1), rotation = 90)

    print('Done')

    # cols = ['Prot.{}'.format(col) for col in range(1, num_cols + 1)]
    # rows = ['Concept # {}'.format(row) for row in range(1, num_rows + 1)]
    #
    # for ax, col in zip(axes[0], cols):
    #     ax.set_title(col)
    #
    # for ax, row in zip(axes[:,0], rows):
    #     ax.set_ylabel(row, rotation=0, size='large')
    #plt.tight_layout()

    if layout == 'vertical':
        fig.subplots_adjust(wspace=0.01, hspace=0.1)
    else:
        fig.subplots_adjust(wspace=0.1, hspace=0.01)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches = 'tight', format='pdf', dpi=300)
        # img = Image.fromarray(np.uint8(fig), 'RGB')
        # img.save('/home/cs16resch11006/ImageNet/scripts/imagenet.png')
    plt.show()
    if return_fig:
        return fig, axes


# define loss function (criterion), optimizer, and learning rate scheduler
criterion = nn.CrossEntropyLoss().cuda(int(args.gpu))

print ("Step 1")

valdir = os.path.join(args.data, 'validation')
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

val_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(valdir, transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])),
    batch_size=args.batch_size, shuffle=False,
    num_workers=args.workers, pin_memory=True)

model = ResNetForImageClassification.from_pretrained("microsoft/resnet-18")
# switch to evaluate mode
model.cuda()
model.eval()

print ("Step 2")

# acc1, all_activs, all_outputs = validate(val_loader, model, criterion, args)
#
# print (acc1)
#
# with open('imagetnet_activs.pkl', 'wb') as f:
#     pickle.dump(all_activs, f)
#
# with open('imagetnet_outputs.pkl', 'wb') as f:
#     pickle.dump(all_outputs, f)

concept_grid(model, val_loader, top_k = 10, save_path = '/om2/user/anirbans/ImageNet/concept_grid_401-512.pdf')

if __name__ == '__main__':
    main()