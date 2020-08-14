import torch.nn.functional as F
import torch
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import torch.optim as optim
import os
import scipy.io as sio

class Gesture(Dataset):
    def __init__(self, train_or_test, transform = None, target_transform = None):
        super(Gesture, self).__init__()
        mat_fname = 'DVS_gesture_100.mat'
        mat_contents = sio.loadmat(mat_fname)
        if train_or_test == 'train':
            self.x_values = mat_contents['train_x_100']
            self.y_values = mat_contents['train_y']
        elif train_or_test == 'test':
            self.x_values = mat_contents['test_x_100']
            self.y_values = mat_contents['test_y']
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        sample = self.x_values[index, :, :]
        # print(sample.shape)
        sample = torch.reshape(torch.tensor(sample), (sample.shape[0],32,32))
        label = self.y_values[index]
        return sample, label

    def __len__(self):
        return len(self.x_values)

if __name__ == '__main__':
    batch_size = 50
    train_dataset = Gesture('train', transform=transforms.ToTensor())
    test_dataset = Gesture('test', transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                        batch_size = batch_size,
                        shuffle = True)

    test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                        batch_size = batch_size,
                        shuffle = False)
    for i, (images, labels) in enumerate(train_loader):
        print(images.shape)
        print(labels)
        # print(i.shape)
        break