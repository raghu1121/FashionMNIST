#!/usr/bin/env python3
import time
start = time.time()
import argparse
import os
import setproctitle
import shutil

import torch
import torch.optim as optim
import torchvision.datasets as dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import densenet
from utils import train,test,adjust_optimizer


torch.cuda.set_device(0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSize', type=int, default=64)
    parser.add_argument('--Epochs', type=int, default=175)
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--save')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--opt', type=str, default='sgd',
                        choices=('sgd', 'adam', 'rmsprop'))
    args = parser.parse_args()

    #Check for cuda
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.save = args.save or 'postprocessing'
    setproctitle.setproctitle(args.save)

    #manual seed on CPU or GPU
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    #Path for saving the progress
    if os.path.exists(args.save):
        shutil.rmtree(args.save)
    os.makedirs(args.save, exist_ok=True)

    # mean and std of the Fashion-MNIST train dataset images
    normMean = [0.2860405969887955]
    normStd = [0.35302424451492237]
    normTransform = transforms.Normalize(normMean, normStd)


    # Transforms : Random crop, random horizontal flip

    trainTransform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normTransform
    ])

    # # Transforms : RandomRotation, RandomVerticalFlip
    # trainTransform = transforms.Compose([
    #     transforms.RandomRotation(90),
    #     transforms.RandomVerticalFlip(),
    #     transforms.ToTensor(),
    #     normTransform
    # ])
    testTransform = transforms.Compose([
        transforms.ToTensor(),
        normTransform
    ])

    #Loading the datasets, if not found will be downloaded
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    loader_train = DataLoader(
        dataset.FashionMNIST(root='Fashion-MNIST', train=True, download=True,
                             transform=trainTransform),
        batch_size=args.batchSize, shuffle=True, **kwargs)
    loader_test = DataLoader(
        dataset.FashionMNIST(root='Fashion-MNIST', train=False, download=True,
                             transform=testTransform),
        batch_size=args.batchSize, shuffle=False, **kwargs)

    # Calling the  Densenet
    dense_net = densenet.DenseNet(growthRate=15, depth=100, reduction=0.5,
                                  bottleneck=True, nClasses=10)


    print('  + Number of params: {}'.format(
        sum([p.data.nelement() for p in dense_net.parameters()])))
    if args.cuda:
        dense_net = dense_net.cuda()
    else:
        print("no cuda")

    #Choosing the optimizer
    if args.opt == 'sgd':
        optimizer = optim.SGD(dense_net.parameters(), lr=1e-1,
                              momentum=0.9, weight_decay=1e-4)
    elif args.opt == 'adam':
        optimizer = optim.Adam(dense_net.parameters(), weight_decay=1e-4)
    elif args.opt == 'rmsprop':
        optimizer = optim.RMSprop(dense_net.parameters(), weight_decay=1e-4)

    #Progress being saved to csv files
    pfile_train = open(os.path.join(args.save, 'train.csv'), 'w')
    pfile_test = open(os.path.join(args.save, 'test.csv'), 'w')

    # running the training loop
    for epoch in range(1, args.Epochs + 1):
        adjust_optimizer(args.opt, optimizer, epoch)
        train(args, epoch, dense_net, loader_train, optimizer, pfile_train)
        test(args, epoch, dense_net, loader_test, pfile_test)
        torch.save(dense_net, os.path.join(args.save, 'latest.pth'))
        os.system('./plot.py {} &'.format(args.save))

    pfile_train.close()
    pfile_test.close()
    end = time.time()
    print(end - start)


if __name__=='__main__':
    main()

