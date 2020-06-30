

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from PIL import Image
import numpy as np


def main():
    mnist_train = datasets.mnist.MNIST(root='mnist/', download=True, train=True, \
                    transform=transforms.ToTensor())
    mnist_test = datasets.mnist.MNIST(root='mnist/', download=True, train=False, \
                    transform=transforms.ToTensor())
    train_dl = DataLoader(mnist_train, batch_size=32, shuffle=True)
    test_dl = DataLoader(mnist_test, batch_size=32, shuffle=False)

    for batch_x, batch_y in train_dl:
        print(batch_x.shape, batch_y.shape)
        break


if __name__ == '__main__':
    main()
