import os
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.utils import save_image
from tensorboardX import SummaryWriter
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import time
import math


class conv_block(nn.Module):
    def __init__(self, in_size, out_size, conv, kernel_size=4, stride=2, padding = 1, activation = "relu", BN = True, bias = False, hook = False):
        super(conv_block, self).__init__()
        self.hook = hook
        if conv == "Conv2d":
            if kernel_size == 3 or kernel_size == 4 :
                padding = 1
            elif kernel_size == 7:
                padding = 3
            else: padding = 0
            if self.hook == False:
                self.conv = nn.Conv2d(in_size, out_size, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
            else:
                self.hook_conv = nn.Conv2d(in_size, out_size, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)

        if conv == "ConvTranspose2d":
            self.conv = nn.ConvTranspose2d(in_size, out_size, kernel_size=kernel_size, stride=stride, padding=padding, bias = bias)
        self.relu = nn.LeakyReLU(0.2, inplace=True) if activation=="LeakyRelu" else nn.ReLU(inplace=True)
        self.BN = nn.BatchNorm2d(out_size) if BN == True else False

    def forward(self, x):
        if self.BN == False:
            if self.hook == True: return self.relu(self.hook_conv(x))
            else: return self.relu(self.conv(x))
        else:
            if self.hook == True: return self.BN(self.relu(self.hook_conv(x)))
            else: return self.BN(self.relu(self.conv(x)))


class AAE_encoder(nn.Module):
    def __init__(self, hidden_size, input_size):
        super(AAE_encoder, self).__init__()
        # Encoder
        self.input_size = input_size                            # 512 512
        self.input_size_sqrt = math.sqrt(input_size)            # 512

        en_act = "LeakyRelu"

        self.en_layer1 = conv_block(3, hidden_size[0], conv = "Conv2d", kernel_size=4, stride=2, activation=en_act)
        self.en_layer2 = conv_block(hidden_size[0], hidden_size[0], conv = "Conv2d", kernel_size=4, stride=2, activation=en_act)
        self.en_layer3 = conv_block(hidden_size[0], hidden_size[1], conv = "Conv2d", kernel_size=4, stride=2, activation=en_act)
        self.en_layer4 = conv_block(hidden_size[1], hidden_size[1], conv = "Conv2d", kernel_size=4, stride=2, activation=en_act)
        self.en_layer5 = conv_block(hidden_size[1], hidden_size[2], conv = "Conv2d", kernel_size=3, stride=1, activation=en_act)
        self.en_layer6 = conv_block(hidden_size[2], hidden_size[2], conv = "Conv2d", kernel_size=4, stride=2, activation=en_act)
        self.en_layer7 = conv_block(hidden_size[2], hidden_size[3], conv = "Conv2d", kernel_size=3, stride=1, activation=en_act)
        self.en_layer8 = conv_block(hidden_size[3], hidden_size[3], conv = "Conv2d", kernel_size=3, stride=1, activation=en_act)
        self.en_conv1 = nn.Conv2d(hidden_size[3], hidden_size[4], kernel_size=8, stride=1, padding=0, bias=False)

    def forward(self, x):
        x = self.en_layer1(x)
        x = self.en_layer2(x)
        x = self.en_layer3(x)
        x = self.en_layer4(x)
        x = self.en_layer5(x)
        x = self.en_layer6(x)
        x = self.en_layer7(x)
        x = self.en_layer8(x)
        return self.en_conv1(x)


class AAE_decoder(nn.Module):
    def __init__(self, hidden_size, input_size):
        super(AAE_decoder, self).__init__()
        # Decoder
        self.input_size = input_size                            # 512 512
        self.input_size_sqrt = math.sqrt(input_size)            # 512

        de_act = "LeakyRelu"

        self.de_layer1 = conv_block(hidden_size[4], hidden_size[3], conv = "ConvTranspose2d", kernel_size=8, stride=1, padding=0, activation=de_act)
        self.de_layer2 = conv_block(hidden_size[3], hidden_size[3], conv = "ConvTranspose2d", kernel_size=3, stride=1, activation=de_act)
        self.de_layer3 = conv_block(hidden_size[3], hidden_size[2], conv = "ConvTranspose2d", kernel_size=3, stride=1, activation=de_act)
        self.de_layer4 = conv_block(hidden_size[2], hidden_size[1], conv = "ConvTranspose2d", activation=de_act)
        self.de_layer5 = conv_block(hidden_size[1], hidden_size[1], conv = "ConvTranspose2d", kernel_size=3, stride=1, activation=de_act)
        self.de_layer6 = conv_block(hidden_size[1], hidden_size[0], conv = "ConvTranspose2d", activation=de_act)
        self.de_layer7 = conv_block(hidden_size[0], hidden_size[0], conv = "ConvTranspose2d", activation=de_act)
        self.de_layer8 = conv_block(hidden_size[0], hidden_size[0], conv = "ConvTranspose2d", activation=de_act)
        self.de_conv = nn.ConvTranspose2d(hidden_size[0], 3, kernel_size=4, stride=2, padding=1, bias=False)


    def forward(self, x):
        x = self.de_layer1(x)
        x = self.de_layer2(x)
        x = self.de_layer3(x)
        x = self.de_layer4(x)
        x = self.de_layer5(x)
        x = self.de_layer6(x)
        x = self.de_layer7(x)
        x = self.de_layer8(x)
        return self.de_conv(x)


class AAE_discriminator(nn.Module):
    def __init__(self, hidden_size, input_size):
        super(AAE_discriminator, self).__init__()
        # Encoder
        self.input_size = input_size                            # 512 512
        self.input_size_sqrt = math.sqrt(input_size)            # 512

        en_act = "LeakyRelu"

        self.dis_layer1 = conv_block(hidden_size[-1], hidden_size[3], conv = "ConvTranspose2d", kernel_size=4, stride=2, activation=en_act)
        self.dis_layer2 = conv_block(hidden_size[3], hidden_size[3], conv = "ConvTranspose2d", kernel_size=4, stride=2, activation=en_act)
        self.dis_layer3 = conv_block(hidden_size[3], hidden_size[2], conv = "ConvTranspose2d", kernel_size=4, stride=2, activation=en_act)
        self.dis_layer4 = conv_block(hidden_size[2], hidden_size[2], conv = "ConvTranspose2d", kernel_size=4, stride=2, activation=en_act)
        self.dis_layer5 = conv_block(hidden_size[2], hidden_size[2], conv = "Conv2d", kernel_size=4, stride=2, activation=en_act)
        self.dis_layer6 = conv_block(hidden_size[2], hidden_size[1], conv = "Conv2d", kernel_size=4, stride=2, activation=en_act)
        self.dis_layer7 = conv_block(hidden_size[1], hidden_size[1], conv = "Conv2d", kernel_size=4, stride=2, activation=en_act)
        self.dis_layer8 = conv_block(hidden_size[1], hidden_size[0], conv = "Conv2d", kernel_size=4, stride=2, activation=en_act)
        self.dis_conv1 = nn.Conv2d(hidden_size[0], 1, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        x = self.dis_layer1(x)
        x = self.dis_layer2(x)
        x = self.dis_layer3(x)
        x = self.dis_layer4(x)
        x = self.dis_layer5(x)
        x = self.dis_layer6(x)
        x = self.dis_layer7(x)
        x = self.dis_layer8(x)
        return self.dis_conv1(x)


# class AAE_encoder(nn.Module):
#     def __init__(self, hidden_size, input_size):
#         super(AAE_encoder, self).__init__()
#         # Encoder
#         self.input_size = input_size                            # 512 512
#         self.input_size_sqrt = math.sqrt(input_size)            # 512
#
#         self.en_layer1 = conv_block(3, hidden_size[0], conv = "Conv2d", kernel_size=7, stride=2, activation="relu")
#         self.en_maxpool1 = nn.MaxPool2d(kernel_size=2)          # 128
#
#         self.en_layer2 = conv_block(hidden_size[0], hidden_size[1], conv = "Conv2d", kernel_size=7, stride=2, activation="relu")
#         self.en_maxpool2 = nn.MaxPool2d(kernel_size=2)          # 32
#
#         self.en_layer3 = conv_block(hidden_size[1], hidden_size[2], conv = "Conv2d", kernel_size=3, stride=2, activation="relu")
#         self.en_maxpool3 = nn.MaxPool2d(kernel_size=2)          # 8
#
#         self.en_layer4 = conv_block(hidden_size[2], hidden_size[3], conv = "Conv2d", kernel_size=3, stride=2, activation="relu")
#         self.en_conv = nn.Conv2d(hidden_size[3], hidden_size[4], kernel_size=3, stride=2, padding=1, bias=False)
#
#     def forward(self, x):
#         x = self.en_maxpool1(self.en_layer1(x))
#         x = self.en_maxpool2(self.en_layer2(x))
#         x = self.en_maxpool3(self.en_layer3(x))
#         x = self.en_layer4(x)
#         return self.en_conv(x)
#
# class AAE_decoder_split(nn.Module):
#     def __init__(self, hidden_size, input_size):
#         super(AAE_decoder_split, self).__init__()
#         # Decoder                                               # 2
#         de_act = "Leakyrelu"
#         self.de_layer0 = conv_block(hidden_size[4], hidden_size[3], conv="ConvTranspose2d", activation=de_act)
#         self.de_layer1 = conv_block(hidden_size[3], hidden_size[2], conv="ConvTranspose2d", activation=de_act)
#         self.de_layer2 = conv_block(hidden_size[2], hidden_size[2], conv="ConvTranspose2d", activation=de_act)
#         self.de_layer3 = conv_block(hidden_size[2], hidden_size[2], conv="ConvTranspose2d", activation=de_act)
#         self.de_layer4 = conv_block(hidden_size[2], hidden_size[1], conv="ConvTranspose2d", activation=de_act)
#         self.de_layer5 = conv_block(hidden_size[1], hidden_size[1], conv="ConvTranspose2d", activation=de_act)
#         self.de_layer6 = conv_block(hidden_size[1], hidden_size[0], conv="ConvTranspose2d", activation=de_act)
#         self.de_layer7 = conv_block(hidden_size[0], 3, conv="ConvTranspose2d", activation=de_act, BN=False)
#
#         self.de_conv0 = nn.Conv2d(3,1,kernel_size=1, stride=1, bias = False)
#         self.de_conv1 = nn.Conv2d(3,1,kernel_size=1, stride=1, bias = False)
#         self.de_conv2 = nn.Conv2d(3,1,kernel_size=1, stride=1, bias = False)
#
#     def forward(self, x):
#         x = self.de_layer0(x)
#         x = self.de_layer1(x)
#         x = self.de_layer2(x)
#         x = self.de_layer3(x)
#         x = self.de_layer4(x)
#         x = self.de_layer5(x)
#         x = self.de_layer6(x)
#         x = self.de_layer7(x)
#
#         return self.de_conv0(x), self.de_conv1(x), self.de_conv2(x)     # for channel split
#
#
# class AAE_discriminator(nn.Module):
#     def __init__(self, hidden_size, input_size):
#         super(AAE_discriminator, self).__init__()
#         # Discriminator             # 4x4
#         dis_act = "relu"
#         self.layer0 = conv_block(hidden_size[4], hidden_size[3], conv="Conv2d", activation=dis_act, kernel_size=7)
#
#         self.layer1 = conv_block(hidden_size[3], hidden_size[2], conv="Conv2d", activation=dis_act, kernel_size=7)
#         self.dis_unpool1 = nn.Upsample(scale_factor=2, mode='nearest')
#
#         self.layer2 = conv_block(hidden_size[2], hidden_size[2], conv="Conv2d", activation=dis_act, kernel_size=7)
#         self.dis_unpool2 = nn.Upsample(scale_factor=2, mode='nearest')
#
#         self.layer3 = conv_block(hidden_size[2], hidden_size[1], conv="Conv2d", activation=dis_act, kernel_size=3, BN=True)
#         self.dis_unpool3 = nn.Upsample(scale_factor=2, mode='nearest')
#         self.dis_conv4 = nn.Conv2d(hidden_size[1], 1, kernel_size=3, stride=2, padding=1, bias=False)  # 4x4
#         self.dis_pool1 = nn.AdaptiveAvgPool2d((1))
#
#
#     def forward(self, x):
#         x = self.layer0(x)
#         x = self.dis_unpool1(self.layer1(x))
#         x = self.dis_unpool2(self.layer2(x))
#         x = self.dis_unpool3(self.layer3(x))
#         return self.dis_pool1(self.dis_conv4(x))
