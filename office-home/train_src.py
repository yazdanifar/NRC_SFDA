import argparse
import os, sys
import os.path as osp
import torchvision
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import network
from torch.utils.data import DataLoader
import random, pdb, math, copy
from tqdm import tqdm
import pickle
from utils import *
from torch import autograd
import shutil

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from sklearn.manifold import TSNE
import pickle as pkl
import matplotlib.cm as cm
import collections
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from sklearn import manifold, datasets
from mpl_toolkits.mplot3d import Axes3D


def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s


def train_source(args):
    dset_loaders = office_load_source(args)
    ## set base network
    netF = network.ResNet_FE().cuda()
    netC = network.feat_classifier(type=args.layer,
                                   class_num=args.class_num,
                                   bottleneck_dim=args.bottleneck).cuda()

    optimizer = optim.SGD([{
        'params': netF.feature_layers.parameters(),
        'lr': args.lr
    }, {
        'params': netF.bottle.parameters(),
        'lr': args.lr * 10
    }, {
        'params': netF.bn.parameters(),
        'lr': args.lr * 10
    }, {
        'params': netC.parameters(),
        'lr': args.lr * 10
    }],
        momentum=0.9,
        weight_decay=5e-4,
        nesterov=True)

    smax = 100

    acc_init = 0
    for epoch in range(args.max_epoch):
        netF.train()
        netC.train()
        iter_source = iter(dset_loaders["source_tr"])
        for batch_idx, (inputs_source, labels_source) in enumerate(iter_source):
            if inputs_source.size(0) == 1:
                continue
            inputs_source, labels_source = inputs_source.cuda(
            ), labels_source.cuda()

            output = netF(inputs_source)
            output = netC(output)

            loss = CrossEntropyLabelSmooth(
                num_classes=args.class_num, epsilon=args.smooth)(
                output, labels_source)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        netF.eval()
        netC.eval()
        acc_s_tr, _ = cal_acc_(dset_loaders['source_te'], netF, netC)
        # acc_s_te, _ = cal_acc_(dset_loaders['source_te'], netF, netB, netC)
        log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(
            args.dset, epoch + 1, args.max_epoch, acc_s_tr * 100)
        args.out_file.write(log_str + '\n')
        args.out_file.flush()
        print(log_str)

        if acc_s_tr >= acc_init:
            acc_init = acc_s_tr
            best_netF = netF.state_dict()
            best_netC = netC.state_dict()
    torch.save(best_netF, osp.join(args.output_dir, "source_F.pt"))
    torch.save(best_netC, osp.join(args.output_dir, "source_C.pt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Domain Adaptation on office-home dataset')
    parser.add_argument('--gpu_id',
                        type=str,
                        nargs='?',
                        default='0',
                        help="device id to run")
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--max_epoch',
                        type=int,
                        default=20,
                        help="maximum epoch")
    parser.add_argument('--batch_size',
                        type=int,
                        default=64,
                        help="batch_size")
    parser.add_argument('--worker',
                        type=int,
                        default=4,
                        help="number of workers")
    parser.add_argument('--dset', type=str, default='a')
    parser.add_argument('--lr',
                        type=float,
                        default=0.001,
                        help="learning rate")
    parser.add_argument('--seed', type=int, default=2021, help="random seed")
    parser.add_argument('--class_num', type=int, default=65)
    parser.add_argument('--par', type=float, default=0.1)
    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--layer',
                        type=str,
                        default="wn",
                        choices=["linear", "wn"])
    parser.add_argument('--classifier',
                        type=str,
                        default="bn",
                        choices=["ori", "bn"])
    parser.add_argument('--smooth', type=float, default=0.1)
    parser.add_argument('--home', action='store_true')
    args = parser.parse_args()
    # args.class_num = 31

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    current_folder = "./runs/source"
    args.output_dir = osp.join(current_folder, 'checkpoint', 'seed' + str(args.seed), args.dset)
    args.log_dir = osp.join(current_folder, 'log', 'seed' + str(args.seed), args.dset)
    for directory in [args.output_dir, args.log_dir]:
        if not osp.exists(directory):
            os.system('mkdir -p ' + directory)
        if not osp.exists(directory):
            os.mkdir(directory)

    if not osp.exists(osp.join(args.output_dir + '/source_F.pt')):
        args.out_file = open(osp.join(args.log_dir, 'log_src_val.txt'), 'w')
        args.out_file.write(print_args(args) + '\n')
        args.out_file.flush()
        train_source(args)
        # test_target(args)
