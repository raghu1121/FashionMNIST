#!/usr/bin/env python3

import numpy as np
import torchvision.datasets as dset
import torchvision.transforms as transforms

data = dset.FashionMNIST(root='Fashion-MNIST', train=True, download=True,
                    transform=transforms.ToTensor()).train_data
data = data.numpy()/255.
means = []
stdevs = []

pixels = data[:,:,:].ravel()
means.append(np.mean(pixels))
stdevs.append(np.std(pixels))

print("means: {}".format(means))
print("stdevs: {}".format(stdevs))
print('transforms.Normalize(mean = {}, std = {})'.format(means, stdevs))
