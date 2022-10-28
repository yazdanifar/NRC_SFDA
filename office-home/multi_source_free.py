import argparse
import os, sys

from torch.utils.tensorboard import SummaryWriter

sys.path.append('./')

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


def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s


def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer


def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer


class ImageList_idx(Dataset):
    def __init__(self,
                 image_list,
                 labels=None,
                 transform=None,
                 target_transform=None,
                 mode='RGB'):
        imgs = make_dataset(image_list, labels)

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        if mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        # for visda
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        return len(self.imgs)


def office_load_idx(args):
    train_bs = args.batch_size
    tt = args.target
    if tt == 'a':
        t = 'Art'
    elif tt == 'c':
        t = 'Clipart'
    elif tt == 'p':
        t = 'Product'
    elif tt == 'r':
        t = 'Real_World'

    t_tr, t_ts = './data/office-home/{}.txt'.format(t), './data/office-home/{}.txt'.format(t)
    prep_dict = {}
    prep_dict['source'] = image_train()
    prep_dict['target'] = image_target()
    prep_dict['test'] = image_test()
    train_target = ImageList_idx(open(t_tr).readlines(),
                                 transform=prep_dict['target'])
    test_target = ImageList_idx(open(t_ts).readlines(), transform=prep_dict['test'])

    dset_loaders = {}
    dset_loaders["target"] = DataLoader(train_target,
                                        batch_size=train_bs,
                                        shuffle=True,
                                        num_workers=args.worker,
                                        drop_last=False)
    dset_loaders["test"] = DataLoader(
        test_target,
        batch_size=train_bs * 3,  # 3
        shuffle=True,
        num_workers=args.worker,
        drop_last=False)
    return dset_loaders


def train_target_separated_views(args, summary):
    dset_loaders = office_load_idx(args)
    ## set base networks
    netF_list = []
    oldC_list = []
    param_list = []
    source_dirs = [osp.join('./runs/source', 'checkpoint', 'seed' + str(args.seed), s)
                   for s in args.source_domains]
    for model_dir in source_dirs:
        netF = network.ResNet_FE().cuda()
        oldC = network.feat_classifier(type=args.layer,
                                       class_num=args.class_num,
                                       bottleneck_dim=args.bottleneck).cuda()

        modelpath = model_dir + '/source_F.pt'
        netF.load_state_dict(torch.load(modelpath))
        modelpath = model_dir + '/source_C.pt'
        oldC.load_state_dict(torch.load(modelpath))
        netF_list.append(netF)
        oldC_list.append(oldC)
        param_list += [
                          {
                              'params': netF.feature_layers.parameters(),
                              'lr': args.lr * .1  # 1
                          },
                          {
                              'params': netF.bottle.parameters(),
                              'lr': args.lr * 1  # 10
                          },
                          {
                              'params': netF.bn.parameters(),
                              'lr': args.lr * 1  # 10
                          },
                          {
                              'params': oldC.parameters(),
                              'lr': args.lr * 1  # 10
                          }
                      ]
    num_srcs = len(netF_list)
    netQ = network.SourceQuantizer(num_srcs).cuda()
    for k, v in netQ.named_parameters():
        param_list += [{'params': v, 'lr': args.alpha_lr}]
    optimizer = optim.SGD(param_list, momentum=0.9, weight_decay=5e-4, nesterov=True)
    optimizer = op_copy(optimizer)
    loader = dset_loaders["target"]
    num_sample = len(loader.dataset)

    for netF, oldC in zip(netF_list, oldC_list):
        netF.eval()
        oldC.eval()
    netQ.eval()

    with torch.no_grad():
        accuracies, _ = cal_acc_multi(dset_loaders["test"], netF_list, oldC_list, netQ)
    for model_id, acc in enumerate(accuracies):
        model_name = args.source_domains[model_id].upper() if model_id < num_srcs else 'Agg'
        log_str = 'Model:{}, Initial accuracy on target:{:.2f}%'.format(model_name, acc * 100)
        args.out_file.write(log_str + '\n')
        print(log_str)
        summary.add_scalar('Accuracy / {}'.format(model_name), acc * 100, 0)
    args.out_file.flush()

    fea_banks = [torch.randn(num_sample, 256) for _ in range(num_srcs)]
    score_bank = torch.randn(num_sample, args.class_num).cuda()
    with torch.no_grad():
        iter_test = iter(loader)
        eye = torch.eye(num_srcs, device='cuda')
        for i in range(len(loader)):
            inputs, _, indx = iter_test.next()
            inputs = inputs.cuda()
            agg_pred = None
            alpha = netQ(eye)
            for model_id, (netF, oldC) in enumerate(zip(netF_list, oldC_list)):
                output = netF.forward(inputs)
                outputs = nn.Softmax(-1)(oldC(output))
                fea_banks[model_id][indx] = F.normalize(output).detach().clone().cpu()
                model_pred = alpha[model_id].repeat(inputs.size(0), 1) * outputs
                agg_pred = model_pred if agg_pred is None else agg_pred + model_pred
            score_bank[indx] = agg_pred.detach().clone()

    max_iter = args.max_epoch * len(dset_loaders["target"])
    interval_iter = max_iter // args.interval
    iter_num = 0

    for netF, oldC in zip(netF_list, oldC_list):
        netF.train()
        oldC.train()
    netQ.train()

    while iter_num < max_iter:

        # comment this if on office-31
        if iter_num > 0.5 * max_iter:
            args.K = 5
            args.KK = 4

        # iter_target = iter(dset_loaders["target"])
        try:
            inputs_target, _, tar_idx = iter_target.next()
        except:
            iter_target = iter(dset_loaders["target"])
            inputs_target, _, tar_idx = iter_target.next()

        if inputs_target.size(0) == 1:
            continue

        iter_num += 1

        # uncomment this if on office-31
        # lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

        inputs_target = inputs_target.cuda()
        agg_pred = None
        alpha = netQ(eye)
        for model_id, (netF, oldC) in enumerate(zip(netF_list, oldC_list)):
            features_test = netF(inputs_target)
            softmax_out = nn.Softmax(dim=1)(oldC(features_test))
            model_pred = alpha[model_id].repeat(inputs_target.size(0), 1) * softmax_out
            agg_pred = model_pred if agg_pred is None else agg_pred + model_pred
            fea_banks[model_id][tar_idx] = F.normalize(features_test).detach().clone().cpu()
        score_bank[tar_idx] = agg_pred.detach().clone()

        loss = torch.tensor(0.0).cuda()
        for model_id in range(num_srcs):
            with torch.no_grad():
                fea_bank = fea_banks[model_id]
                output_f_ = fea_bank[tar_idx]
                distance = output_f_ @ fea_bank.T
                _, idx_near = torch.topk(distance,
                                         dim=-1,
                                         largest=True,
                                         k=args.K + 1)
                idx_near = idx_near[:, 1:]  # batch x K
                score_near = score_bank[idx_near]  # batch x K x C

                fea_near = fea_bank[idx_near]  # batch x K x num_dim
                fea_bank_re = fea_bank.unsqueeze(0).expand(fea_near.shape[0], -1,
                                                           -1)  # batch x n x dim
                distance_ = torch.bmm(fea_near, fea_bank_re.permute(0, 2, 1))  # batch x K x n
                _, idx_near_near = torch.topk(
                    distance_, dim=-1, largest=True,
                    k=args.KK + 1)  # M near neighbors for each of above K ones
                idx_near_near = idx_near_near[:, :, 1:]  # batch x K x M
                tar_idx_ = tar_idx.unsqueeze(-1).unsqueeze(-1)
                match = (idx_near_near == tar_idx_).sum(-1).float()  # batch x K
                weight = torch.where(
                    match > 0., match,
                    torch.ones_like(match).fill_(0.1))  # batch x K

                weight_kk = weight.unsqueeze(-1).expand(-1, -1,
                                                        args.KK)  # batch x K x M

                # weight_kk[idx_near_near == tar_idx_] = 0

                score_near_kk = score_bank[idx_near_near]  # batch x K x M x C
                # print(weight_kk.shape)
                weight_kk = weight_kk.contiguous().view(weight_kk.shape[0],
                                                        -1)  # batch x KM
                weight_kk = weight_kk.fill_(0.1)
                score_near_kk = score_near_kk.contiguous().view(
                    score_near_kk.shape[0], -1, args.class_num)  # batch x KM x C

            # nn of nn
            output_re = agg_pred.unsqueeze(1).expand(-1, args.K * args.KK, -1)  # batch x KM x C
            const = torch.mean(
                (F.kl_div(output_re, score_near_kk, reduction='none').sum(-1) * weight_kk.cuda()).sum(1))
            loss += alpha[model_id].detach().squeeze() * torch.mean(const)

            # nn
            pred_un = agg_pred.unsqueeze(1).expand(-1, args.K, -1)  # batch x K x C

            loss += alpha[model_id].detach().squeeze() * \
                    torch.mean((F.kl_div(pred_un, score_near, reduction='none').sum(-1) * weight.cuda()).sum(1))

        msoftmax = agg_pred.mean(dim=0)
        im_div = torch.sum(msoftmax * torch.log(msoftmax + 1e-5))
        loss += im_div
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if iter_num % interval_iter == 0 or iter_num == max_iter:
            for netF, oldC in zip(netF_list, oldC_list):
                netF.eval()
                oldC.eval()
            netQ.eval()
            alpha = netQ(eye).detach().clone().cpu()
            for i, al in enumerate(alpha):
                m = args.source_domains[i].upper()
                summary.add_scalar('Alpha / {}'.format(m), al, iter_num)
            # noinspection DuplicatedCode
            accuracies, _ = cal_acc_multi(dset_loaders['test'], netF_list, oldC_list, netQ)
            for model_id, acc in enumerate(accuracies):
                model_name = args.source_domains[model_id].upper() if model_id < num_srcs else 'Agg'
                log_str = 'Iter:{}/{}; Model:{}, Accuracy on target:{:.2f}%'.format(
                    iter_num, max_iter, model_name, acc * 100)
                args.out_file.write(log_str + '\n')
                print(log_str)
                summary.add_scalar('Accuracy / {}'.format(model_name), acc * 100, iter_num)
            args.out_file.flush()

            for netF, oldC in zip(netF_list, oldC_list):
                netF.train()
                oldC.train()
            netQ.train()
    summary.flush()

def train_target_shared_views(args, summary):
    dset_loaders = office_load_idx(args)
    ## set base networks
    netF_list = []
    oldC_list = []
    param_list = []
    source_dirs = [osp.join('./runs/source', 'checkpoint', 'seed' + str(args.seed), s)
                   for s in args.source_domains]
    for model_dir in source_dirs:
        netF = network.ResNet_FE().cuda()
        oldC = network.feat_classifier(type=args.layer,
                                       class_num=args.class_num,
                                       bottleneck_dim=args.bottleneck).cuda()

        modelpath = model_dir + '/source_F.pt'
        netF.load_state_dict(torch.load(modelpath))
        modelpath = model_dir + '/source_C.pt'
        oldC.load_state_dict(torch.load(modelpath))
        netF_list.append(netF)
        oldC_list.append(oldC)
        param_list += [
                          {
                              'params': netF.feature_layers.parameters(),
                              'lr': args.lr * .1  # 1
                          },
                          {
                              'params': netF.bottle.parameters(),
                              'lr': args.lr * 1  # 10
                          },
                          {
                              'params': netF.bn.parameters(),
                              'lr': args.lr * 1  # 10
                          },
                          {
                              'params': oldC.parameters(),
                              'lr': args.lr * 1  # 10
                          }
                      ]
    num_srcs = len(netF_list)
    netQ = network.SourceQuantizer(num_srcs).cuda()
    for k, v in netQ.named_parameters():
        param_list += [{'params': v, 'lr': args.alpha_lr}]
    optimizer = optim.SGD(param_list, momentum=0.9, weight_decay=5e-4, nesterov=True)
    optimizer = op_copy(optimizer)
    loader = dset_loaders["target"]
    num_sample = len(loader.dataset)

    for netF, oldC in zip(netF_list, oldC_list):
        netF.eval()
        oldC.eval()
    netQ.eval()

    with torch.no_grad():
        accuracies, _ = cal_acc_multi(dset_loaders["test"], netF_list, oldC_list, netQ)
    for model_id, acc in enumerate(accuracies):
        model_name = args.source_domains[model_id].upper() if model_id < num_srcs else 'Agg'
        log_str = 'Model:{}, Initial accuracy on target:{:.2f}%'.format(model_name, acc * 100)
        args.out_file.write(log_str + '\n')
        print(log_str)
        summary.add_scalar('Accuracy / {}'.format(model_name), acc * 100, 0)
    args.out_file.flush()

    fea_bank = [torch.randn(num_sample, 256)]
    score_bank = torch.randn(num_sample, args.class_num).cuda()
    with torch.no_grad():
        iter_test = iter(loader)
        eye = torch.eye(num_srcs, device='cuda')
        for i in range(len(loader)):
            inputs, _, indx = iter_test.next()
            inputs = inputs.cuda()
            agg_pred = None
            agg_feat = None
            alpha = netQ(eye)
            for model_id, (netF, oldC) in enumerate(zip(netF_list, oldC_list)):
                output = netF.forward(inputs)
                outputs = nn.Softmax(-1)(oldC(output))
                coeff = alpha[model_id].repeat(inputs.size(0), 1)
                norm_feat = F.normalize(output).detach()
                norm_feat = coeff * norm_feat
                model_pred = coeff * outputs
                agg_pred = model_pred if agg_pred is None else agg_pred + model_pred
                agg_feat = norm_feat if agg_feat is None else agg_feat + norm_feat
            score_bank[indx] = agg_pred.detach().clone()
            fea_bank[indx] = agg_feat.detach().clone().cpu()

    max_iter = args.max_epoch * len(dset_loaders["target"])
    interval_iter = max_iter // args.interval
    iter_num = 0

    for netF, oldC in zip(netF_list, oldC_list):
        netF.train()
        oldC.train()
    netQ.train()

    while iter_num < max_iter:

        # comment this if on office-31
        if iter_num > 0.5 * max_iter:
            args.K = 5
            args.KK = 4

        # iter_target = iter(dset_loaders["target"])
        try:
            inputs_target, _, tar_idx = iter_target.next()
        except:
            iter_target = iter(dset_loaders["target"])
            inputs_target, _, tar_idx = iter_target.next()

        if inputs_target.size(0) == 1:
            continue

        iter_num += 1

        # uncomment this if on office-31
        # lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

        inputs_target = inputs_target.cuda()
        agg_pred = None
        agg_feat = None
        alpha = netQ(eye)
        for model_id, (netF, oldC) in enumerate(zip(netF_list, oldC_list)):
            output = netF.forward(inputs_target)
            outputs = nn.Softmax(-1)(oldC(output))
            coeff = alpha[model_id].repeat(inputs_target.size(0), 1)
            norm_feat = F.normalize(output).detach()
            norm_feat = coeff.detach() * norm_feat
            model_pred = coeff * outputs
            agg_pred = model_pred if agg_pred is None else agg_pred + model_pred
            agg_feat = norm_feat if agg_feat is None else agg_feat + norm_feat
        score_bank[indx] = agg_pred.detach().clone()
        fea_bank[indx] = agg_feat.detach().clone().cpu()

        with torch.no_grad():
            output_f_ = fea_bank[tar_idx]
            distance = output_f_ @ fea_bank.T
            _, idx_near = torch.topk(distance,
                                     dim=-1,
                                     largest=True,
                                     k=args.K + 1)
            idx_near = idx_near[:, 1:]  # batch x K
            score_near = score_bank[idx_near]  # batch x K x C

            fea_near = fea_bank[idx_near]  # batch x K x num_dim
            fea_bank_re = fea_bank.unsqueeze(0).expand(fea_near.shape[0], -1, -1)  # batch x n x dim
            distance_ = torch.bmm(fea_near, fea_bank_re.permute(0, 2, 1))  # batch x K x n
            _, idx_near_near = torch.topk(
                distance_, dim=-1, largest=True,
                k=args.KK + 1)  # M near neighbors for each of above K ones
            idx_near_near = idx_near_near[:, :, 1:]  # batch x K x M
            tar_idx_ = tar_idx.unsqueeze(-1).unsqueeze(-1)
            match = (idx_near_near == tar_idx_).sum(-1).float()  # batch x K
            weight = torch.where(
                match > 0., match,
                torch.ones_like(match).fill_(0.1))  # batch x K

            weight_kk = weight.unsqueeze(-1).expand(-1, -1, args.KK)  # batch x K x M

            # weight_kk[idx_near_near == tar_idx_] = 0

            score_near_kk = score_bank[idx_near_near]  # batch x K x M x C
            # print(weight_kk.shape)
            weight_kk = weight_kk.contiguous().view(weight_kk.shape[0],
                                                    -1)  # batch x KM
            weight_kk = weight_kk.fill_(0.1)
            score_near_kk = score_near_kk.contiguous().view(
                score_near_kk.shape[0], -1, args.class_num)  # batch x KM x C

        # nn of nn
        output_re = agg_pred.unsqueeze(1).expand(-1, args.K * args.KK, -1)  # batch x KM x C
        const = torch.mean(
            (F.kl_div(output_re, score_near_kk, reduction='none').sum(-1) * weight_kk.cuda()).sum(1))
        loss = torch.mean(const)

        # nn
        pred_un = agg_pred.unsqueeze(1).expand(-1, args.K, -1)  # batch x K x C

        loss += torch.mean((F.kl_div(pred_un, score_near, reduction='none').sum(-1) * weight.cuda()).sum(1))

        msoftmax = agg_pred.mean(dim=0)
        im_div = torch.sum(msoftmax * torch.log(msoftmax + 1e-5))
        loss += im_div
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if iter_num % interval_iter == 0 or iter_num == max_iter:
            for netF, oldC in zip(netF_list, oldC_list):
                netF.eval()
                oldC.eval()
            netQ.eval()
            alpha = netQ(eye).detach().clone().cpu()
            for i, al in enumerate(alpha):
                m = args.source_domains[i].upper()
                summary.add_scalar('Alpha / {}'.format(m), al, iter_num)
            # noinspection DuplicatedCode
            accuracies, _ = cal_acc_multi(dset_loaders['test'], netF_list, oldC_list, netQ)
            for model_id, acc in enumerate(accuracies):
                model_name = args.source_domains[model_id].upper() if model_id < num_srcs else 'Agg'
                log_str = 'Iter:{}/{}; Model:{}, Accuracy on target:{:.2f}%'.format(
                    iter_num, max_iter, model_name, acc * 100)
                args.out_file.write(log_str + '\n')
                print(log_str)
                summary.add_scalar('Accuracy / {}'.format(model_name), acc * 100, iter_num)
            args.out_file.flush()

            for netF, oldC in zip(netF_list, oldC_list):
                netF.train()
                oldC.train()
            netQ.train()
    summary.flush()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Domain Adaptation on office-home dataset')
    parser.add_argument('--gpu_id',
                        type=str,
                        nargs='?',
                        default='0',
                        help="device id to run")
    parser.add_argument('--separate_views',
                        type=bool,
                        default=True,
                        help="to sum different views or not")
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--max_epoch',
                        type=int,
                        default=40,
                        help="maximum epoch")  # set to 50 on office-31
    parser.add_argument('--batch_size',
                        type=int,
                        default=64,
                        help="batch_size")
    parser.add_argument('--interval', type=int, default=15)
    parser.add_argument('--worker',
                        type=int,
                        default=4,
                        help="number of workers")
    parser.add_argument('--k',
                        type=int,
                        default=2,
                        help="number of neighborhoods")
    parser.add_argument('--target', type=str, default='p')
    parser.add_argument('--choice', type=str, default='shot')
    parser.add_argument('--lr',
                        type=float,
                        default=0.001,
                        help="learning rate")
    parser.add_argument('--alpha_lr',
                        type=float,
                        default=0.001,
                        help="alpha learning rate")
    parser.add_argument('--seed', type=int, default=2021, help="random seed")
    parser.add_argument('--class_num', type=int, default=65)
    parser.add_argument('--K', type=int, default=4)  # set to 2 on office-31 (or 3 on a2w)
    parser.add_argument('--KK', type=int, default=3)  # set to 3 on office-31 (or 2 on a2w)
    parser.add_argument('--alpha', type=float, default=0.5)
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
    parser.add_argument('--output', type=str, default='weight')  # trainingC_2
    parser.add_argument('--file', type=str, default='log')
    parser.add_argument('--home', action='store_true')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.backends.cudnn.deterministic = True

    current_folder = "./runs/target"
    postfix = '_' + args.exp_name if len(args.exp_name) > 0 else ''
    args.output_dir = osp.join(current_folder, 'checkpoint', 'seed' + str(args.seed), args.dset + postfix)
    args.log_dir = osp.join(current_folder, 'log', 'seed' + str(args.seed), args.dset + postfix)
    for directory in [args.output_dir, args.log_dir]:
        if not osp.exists(directory):
            os.system('mkdir -p ' + directory)
    args.out_file = open(osp.join(args.log_dir, 'log.txt'), 'w')
    args.out_file.write(print_args(args) + '\n')
    args.out_file.flush()
    summary = SummaryWriter(args.log_dir)
    args.source_domains = [s for s in ['a', 'c', 'p', 'r'] if s != args.target]
    if args.separate_views:
        train_target_separated_views(args, summary)
    else:
        train_target_shared_views(args, summary)
