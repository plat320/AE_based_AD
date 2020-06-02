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

class AAE(nn.Module):                                           # for MNIST data
    def __init__(self, hidden_size, input_size):
        super(AAE, self).__init__()
        # Encoder
        self.input_size = input_size
        self.fc1 = nn.Linear(self.input_size, hidden_size[0])
        self.en_relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_size[0],hidden_size[1])
        self.en_relu2 = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(hidden_size[1], hidden_size[2])

        # Decoder
        self.de_fc1 = nn.Linear(hidden_size[2], hidden_size[1])
        self.de_relu1 = nn.ReLU(inplace=True)
        self.de_fc2 = nn.Linear(hidden_size[1], hidden_size[0])
        self.de_relu2 = nn.ReLU(inplace=True)
        self.de_fc3 = nn.Linear(hidden_size[0], input_size)

        # Discriminator
        self.dis_fc1 = nn.Linear(hidden_size[2], hidden_size[1])
        self.dis_relu = nn.ReLU(inplace=True)
        self.dis_fc2 = nn.Linear(hidden_size[1], 1)

    def Encoder(self, x):
        x = self.en_relu1(self.fc1(x))
        x = self.en_relu2(self.fc2(x))
        return self.fc3(x)

    def Decoder(self, x):
        x = self.de_relu1(self.de_fc1(x))
        x = self.de_relu2(self.de_fc2(x))
        return self.de_fc3(x)

    def Discriminator(self, x):
        x = self.dis_relu(self.dis_fc1(x))
        return self.dis_fc2(x)

    def forward(self, x):
        x = self.Encoder(x)
        y = self.Decoder(x)
        z = self.Discriminator(x)
        return x,y,z


class AAE_cifar(nn.Module):                                           # for MNIST data
    def __init__(self, hidden_size, input_size):
        super(AAE_cifar, self).__init__()
        # Encoder
        self.input_size = input_size
        self.fc1 = nn.Linear(self.input_size, hidden_size[0])
        self.en_relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_size[0],hidden_size[1])
        self.en_relu2 = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(hidden_size[1], hidden_size[2])
        self.en_relu3 = nn.ReLU(inplace=True)
        self.fc4 = nn.Linear(hidden_size[2], hidden_size[3])

        # Decoder
        self.de_fc1 = nn.Linear(hidden_size[3], hidden_size[2])
        self.de_relu1 = nn.ReLU(inplace=True)
        self.de_fc2 = nn.Linear(hidden_size[2], hidden_size[1])
        self.de_relu2 = nn.ReLU(inplace=True)
        self.de_fc3 = nn.Linear(hidden_size[1], hidden_size[0])
        self.de_relu3 = nn.ReLU(inplace=True)
        self.de_fc4 = nn.Linear(hidden_size[0], input_size)

        # Discriminator
        self.dis_fc1 = nn.Linear(hidden_size[3], hidden_size[2])
        self.dis_relu1 = nn.ReLU(inplace=True)
        self.dis_fc2 = nn.Linear(hidden_size[2], hidden_size[1])
        self.dis_relu2 = nn.ReLU(inplace=True)
        self.dis_fc3 = nn.Linear(hidden_size[1], 1)

    def Encoder(self, x):
        x = self.en_relu1(self.fc1(x))
        x = self.en_relu2(self.fc2(x))
        x = self.en_relu3(self.fc3(x))
        return self.fc4(x)

    def Decoder(self, x):
        x = self.de_relu1(self.de_fc1(x))
        x = self.de_relu2(self.de_fc2(x))
        x = self.de_relu3(self.de_fc3(x))
        return self.de_fc4(x)

    def Discriminator(self, x):
        x = self.dis_relu1(self.dis_fc1(x))
        x = self.dis_relu2(self.dis_fc2(x))
        return self.dis_fc3(x)

    def forward(self, x):
        x = self.Encoder(x)
        y = self.Decoder(x)
        z = self.Discriminator(x)
        return x,y,z

class AAE_encoder(nn.Module):
    def __init__(self, hidden_size, input_size):
        super(AAE_encoder, self).__init__()
        # Encoder
        self.hidden_channel = 4
        self.input_size = input_size                            # 32 32
        self.input_size_sqrt = math.sqrt(input_size)            # 32

        self.en_conv1 = nn.Conv2d(3, hidden_size[0], kernel_size=3, stride=2, padding=1, bias=False)
        self.en_relu1 = nn.ReLU(inplace=True)                   # 16x16
        self.en_bn1 = nn.BatchNorm2d(hidden_size[0])
        self.en_conv2 = nn.Conv2d(hidden_size[0], hidden_size[1], kernel_size=3, stride=2, padding=1, bias=False)
        self.en_relu2 = nn.ReLU(inplace=True)                   # 8x8
        self.en_bn2 = nn.BatchNorm2d(hidden_size[1])
        self.en_conv3 = nn.Conv2d(hidden_size[1], hidden_size[2], kernel_size=3, stride=2, padding=1, bias=False)
        self.en_relu3 = nn.ReLU(inplace=True)                   # 4x4
        self.en_bn3 = nn.BatchNorm2d(hidden_size[2])
        self.en_conv4 = nn.Conv2d(hidden_size[2], hidden_size[3], kernel_size=1, stride=1, bias=False)   # 4x4


    def forward(self, x):
        x = self.en_bn1(self.en_relu1(self.en_conv1(x)))
        x = self.en_bn2(self.en_relu2(self.en_conv2(x)))
        x = self.en_bn3(self.en_relu3(self.en_conv3(x)))
        return self.en_conv4(x)

class AAE_decoder_split(nn.Module):
    def __init__(self, hidden_size, input_size):
        super(AAE_decoder_split, self).__init__()
        # Decoder
        self.de_conv0 = nn.ConvTranspose2d(hidden_size[3], hidden_size[2], kernel_size=4, stride=2, padding = 1, bias = False)
        self.de_relu0 = nn.ReLU(inplace=True)
        self.de_bn0 = nn.BatchNorm2d(hidden_size[2])
        self.de_conv1 = nn.Conv2d(hidden_size[2], hidden_size[1], kernel_size=1, stride=1, bias=False)
        self.de_relu1 = nn.ReLU(inplace=True)
        self.de_bn1 = nn.BatchNorm2d(hidden_size[1])
        self.de_conv2 = nn.ConvTranspose2d(hidden_size[1], hidden_size[1], kernel_size=4, stride=2, padding = 1, bias = False)
        self.de_relu2 = nn.ReLU(inplace=True)
        self.de_bn2 = nn.BatchNorm2d(hidden_size[1])
        self.de_conv3 = nn.Conv2d(hidden_size[1], hidden_size[0], kernel_size=1, stride=1, bias=False)
        self.de_relu3 = nn.ReLU(inplace=True)
        self.de_bn3 = nn.BatchNorm2d(hidden_size[0])
        self.de_conv4 = nn.ConvTranspose2d(hidden_size[0], 3, kernel_size=4, stride=2, padding=1, bias=False)
        self.de_relu4 = nn.ReLU(inplace=True)
        self.de_conv5 = nn.Conv2d(3,3,kernel_size=1, stride=1, bias = False)
        self.de_conv5_0 = nn.Conv2d(3,1,kernel_size=1, stride=1, bias = False)
        self.de_conv5_1 = nn.Conv2d(3,1,kernel_size=1, stride=1, bias = False)
        self.de_conv5_2 = nn.Conv2d(3,1,kernel_size=1, stride=1, bias = False)

    def forward(self, x):
        x = self.de_bn0(self.de_relu0(self.de_conv0(x)))
        x = self.de_bn1(self.de_relu1(self.de_conv1(x)))
        x = self.de_bn2(self.de_relu2(self.de_conv2(x)))
        x = self.de_bn3(self.de_relu3(self.de_conv3(x)))
        x = self.de_relu4(self.de_conv4(x))
        # return self.de_conv5(x)
        return self.de_conv5_0(x), self.de_conv5_1(x), self.de_conv5_2(x)     # for channel split


class AAE_decoder(nn.Module):
    def __init__(self, hidden_size, input_size):
        super(AAE_decoder, self).__init__()
        # Decoder
        self.de_conv0 = nn.ConvTranspose2d(hidden_size[3], hidden_size[2], kernel_size=4, stride=2, padding = 1, bias = False)
        self.de_relu0 = nn.ReLU(inplace=True)
        self.de_bn0 = nn.BatchNorm2d(hidden_size[2])
        self.de_conv1 = nn.Conv2d(hidden_size[2], hidden_size[1], kernel_size=1, stride=1, bias=False)
        self.de_relu1 = nn.ReLU(inplace=True)
        self.de_bn1 = nn.BatchNorm2d(hidden_size[1])
        self.de_conv2 = nn.ConvTranspose2d(hidden_size[1], hidden_size[1], kernel_size=4, stride=2, padding = 1, bias = False)
        self.de_relu2 = nn.ReLU(inplace=True)
        self.de_bn2 = nn.BatchNorm2d(hidden_size[1])
        self.de_conv3 = nn.Conv2d(hidden_size[1], hidden_size[0], kernel_size=1, stride=1, bias=False)
        self.de_relu3 = nn.ReLU(inplace=True)
        self.de_bn3 = nn.BatchNorm2d(hidden_size[0])
        self.de_conv4 = nn.ConvTranspose2d(hidden_size[0], 3, kernel_size=4, stride=2, padding=1, bias=False)
        self.de_relu4 = nn.ReLU(inplace=True)
        self.de_conv5 = nn.Conv2d(3,3,kernel_size=1, stride=1, bias = False)
        self.de_conv5_0 = nn.Conv2d(3,1,kernel_size=1, stride=1)
        self.de_conv5_1 = nn.Conv2d(3,1,kernel_size=1, stride=1)
        self.de_conv5_2 = nn.Conv2d(3,1,kernel_size=1, stride=1)

    def forward(self, x):
        x = self.de_bn0(self.de_relu0(self.de_conv0(x)))
        x = self.de_bn1(self.de_relu1(self.de_conv1(x)))
        x = self.de_bn2(self.de_relu2(self.de_conv2(x)))
        x = self.de_bn3(self.de_relu3(self.de_conv3(x)))
        x = self.de_relu4(self.de_conv4(x))
        return self.de_conv5(x)
        # return self.de_conv5_0(x), self.de_conv5_1(x), self.de_conv5_2(x)     # for channel split



class AAE_discriminator(nn.Module):
    def __init__(self, hidden_size, input_size):
        super(AAE_discriminator, self).__init__()
        # Discriminator             # 4x4
        self.dis_conv0 = nn.Conv2d(hidden_size[3], hidden_size[2], kernel_size=1, stride=1, bias=False)
        self.dis_relu0 = nn.ReLU(inplace=True)  # 4x4
        self.dis_bn0 = nn.BatchNorm2d(hidden_size[2])
        self.dis_conv1 = nn.Conv2d(hidden_size[2], hidden_size[2], kernel_size=3, stride=2, padding=1, bias=False)
        self.dis_relu1 = nn.ReLU(inplace=True)                  # 4x4
        self.dis_bn1 = nn.BatchNorm2d(hidden_size[2])
        self.dis_conv2 = nn.Conv2d(hidden_size[2], hidden_size[1], kernel_size=3, stride=2, padding=1, bias=False)
        self.dis_relu2 = nn.ReLU(inplace=True)                  # 2x2
        self.dis_bn2 = nn.BatchNorm2d(hidden_size[1])
        self.dis_conv3 = nn.Conv2d(hidden_size[1], 1, kernel_size=3, stride=2, padding=1, bias=False)   # 4x4
        self.dis_sigmoid = nn.Sigmoid()                         # 1x1
        self.dis_pool1 = nn.AdaptiveAvgPool2d((1))

    def forward(self, x):
        x = self.dis_bn0(self.dis_relu0(self.dis_conv0(x)))
        x = self.dis_bn1(self.dis_relu1(self.dis_conv1(x)))
        x = self.dis_bn2(self.dis_relu2(self.dis_conv2(x)))
        return self.dis_conv3(x)





class AAE_conv(nn.Module):
    def __init__(self, hidden_size, input_size):
        super(AAE_conv, self).__init__()
        # Encoder
        self.hidden_channel = 4
        self.input_size = input_size                            # 32 32
        self.input_size_sqrt = math.sqrt(input_size)            # 32

        self.en_conv1 = nn.Conv2d(3, hidden_size[0], kernel_size=3, stride=2, padding=1)
        self.en_relu1 = nn.ReLU(inplace=True)                   # 16x16
        self.en_bn1 = nn.BatchNorm2d(hidden_size[0])
        self.en_conv2 = nn.Conv2d(hidden_size[0], hidden_size[1], kernel_size=3, stride=2, padding=1)
        self.en_relu2 = nn.ReLU(inplace=True)                   # 8x8
        self.en_bn2 = nn.BatchNorm2d(hidden_size[1])
        self.en_conv3 = nn.Conv2d(hidden_size[1], hidden_size[2], kernel_size=3, stride=2, padding=1)
        self.en_relu3 = nn.ReLU(inplace=True)                   # 4x4
        self.en_bn3 = nn.BatchNorm2d(hidden_size[2])
        self.en_conv4 = nn.Conv2d(hidden_size[2], hidden_size[3], kernel_size=1, stride=1)   # 4x4


        # Decoder
        self.de_conv0 = nn.ConvTranspose2d(hidden_size[3], hidden_size[2], kernel_size=4, stride=2, padding = 1, bias = False)
        self.de_relu0 = nn.ReLU(inplace=True)
        self.de_bn0 = nn.BatchNorm2d(hidden_size[2])
        self.de_conv1 = nn.Conv2d(hidden_size[2], hidden_size[1], kernel_size=1, stride=1)
        self.de_relu1 = nn.ReLU(inplace=True)
        self.de_bn1 = nn.BatchNorm2d(hidden_size[1])
        self.de_conv2 = nn.ConvTranspose2d(hidden_size[1], hidden_size[1], kernel_size=4, stride=2, padding = 1, bias = False)
        self.de_relu2 = nn.ReLU(inplace=True)
        self.de_bn2 = nn.BatchNorm2d(hidden_size[1])
        self.de_conv3 = nn.Conv2d(hidden_size[1], hidden_size[0], kernel_size=1, stride=1)
        self.de_relu3 = nn.ReLU(inplace=True)
        self.de_bn3 = nn.BatchNorm2d(hidden_size[0])
        self.de_conv4 = nn.ConvTranspose2d(hidden_size[0], 3, kernel_size=4, stride=2, padding=1, bias=False)
        self.de_relu4 = nn.ReLU(inplace=True)
        self.de_conv5 = nn.Conv2d(3,3,kernel_size=1, stride=1)



        # Discriminator             # 4x4
        self.dis_conv0 = nn.Conv2d(hidden_size[3], hidden_size[2], kernel_size=1, stride=1)
        self.dis_relu0 = nn.ReLU(inplace=True)  # 4x4
        self.dis_bn0 = nn.BatchNorm2d(hidden_size[2])
        self.dis_conv1 = nn.Conv2d(hidden_size[2], hidden_size[2], kernel_size=3, stride=2, padding=1)
        self.dis_relu1 = nn.ReLU(inplace=True)                  # 4x4
        self.dis_bn1 = nn.BatchNorm2d(hidden_size[2])
        self.dis_conv2 = nn.Conv2d(hidden_size[2], hidden_size[1], kernel_size=3, stride=2, padding=1)
        self.dis_relu2 = nn.ReLU(inplace=True)                  # 2x2
        self.dis_bn2 = nn.BatchNorm2d(hidden_size[1])
        self.dis_conv3 = nn.Conv2d(hidden_size[1], 1, kernel_size=3, stride=2, padding=1)   # 4x4
        self.dis_sigmoid = nn.Sigmoid()                         # 1x1
        self.dis_pool1 = nn.AdaptiveAvgPool2d((1))

    def Encoder(self, x):
        x = self.en_bn1(self.en_relu1(self.en_conv1(x)))
        x = self.en_bn2(self.en_relu2(self.en_conv2(x)))
        x = self.en_bn3(self.en_relu3(self.en_conv3(x)))
        return self.en_conv4(x)

    def Decoder(self, x):
        x = self.de_bn0(self.de_relu0(self.de_conv0(x)))
        x = self.de_bn1(self.de_relu1(self.de_conv1(x)))
        x = self.de_bn2(self.de_relu2(self.de_conv2(x)))
        x = self.de_bn3(self.de_relu3(self.de_conv3(x)))
        x = self.de_relu4(self.de_conv4(x))
        return self.de_conv5(x)

    def Discriminator(self, x):
        x = self.dis_bn0(self.dis_relu0(self.dis_conv0(x)))
        x = self.dis_bn1(self.dis_relu1(self.dis_conv1(x)))
        x = self.dis_bn2(self.dis_relu2(self.dis_conv2(x)))
        return self.dis_conv3(x)

    def forward(self, x):
        x = self.Encoder(x)
        y = self.Decoder(x)
        z = self.Discriminator(x)
        return x, y, z


class VAE(nn.Module):
    def __init__(self, hidden_size, input_size):
        super(VAE, self).__init__()
        self.input_size = input_size
        self.fc1 = nn.Linear(self.input_size, hidden_size[0])
        self.fc2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.fc2.weight.data = weight.clone()
        self.fc3_1 = nn.Linear(hidden_size[1], hidden_size[2])
        self.fc3_2 = nn.Linear(hidden_size[1], hidden_size[2])
        self.fc4 = nn.Linear(hidden_size[2], hidden_size[1])
        self.fc5 = nn.Linear(hidden_size[1], hidden_size[0])
        self.fc6 = nn.Linear(hidden_size[0], self.input_size)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def encoder(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3_1(x), self.fc3_2(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(logvar / 2)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decoder(self, x, a):
        x = self.relu(self.fc4(x))
        x = self.relu(self.fc5(x))
        x = self.fc6(x)
        return x

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        z = self.decoder(z, 1)
        return z, mu, log_var


class VAE_conv(nn.Module):
    def __init__(self, hidden_size, input_size):
        super(VAE_conv, self).__init__()
        self.input_size = int(math.sqrt(input_size))

        self.conv1 = nn.Conv2d(1, hidden_size[0], kernel_size=(3,3), stride=(2,2), padding=(1,1))
        self.bn1 = nn.BatchNorm2d(hidden_size[0])
        self.relu1 = nn.ReLU(inplace=True)
        self.lrelu1 = nn.LeakyReLU(0.2, inplace=True)
        self.dropout1 = nn.Dropout(0.2)
        self.conv2 = nn.Conv2d(hidden_size[0], hidden_size[1], kernel_size=(3,3), stride=(2,2), padding=(1,1))
        self.bn2 = nn.BatchNorm2d(hidden_size[1])
        self.relu2 = nn.ReLU(inplace=True)
        self.lrelu2 = nn.LeakyReLU(0.2, inplace=True)
        self.dropout2 = nn.Dropout(0.2)
        self.conv3 = nn.Conv2d(hidden_size[1], hidden_size[2], kernel_size=(1,1), stride=(1,1))
        self.bn3 = nn.BatchNorm2d(hidden_size[2])
        self.relu3 = nn.ReLU(inplace=True)
        self.lrelu3 = nn.LeakyReLU(0.2, inplace=True)
        self.dropout3 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(int(self.input_size ** 2 * hidden_size[2] / 16), hidden_size[2])
        self.bn4 = nn.BatchNorm1d(hidden_size[2])
        self.relu4 = nn.ReLU(inplace=True)
        self.lrelu4 = nn.LeakyReLU(0.2, inplace=True)
        self.fc2_1 = nn.Linear(hidden_size[2], hidden_size[3])
        self.fc2_2 = nn.Linear(hidden_size[2], hidden_size[3])


        self.fc3 = nn.Linear(hidden_size[3], hidden_size[2]*64)
        self.derelu1 = nn.ReLU(inplace=True)
        self.delrelu1 = nn.LeakyReLU(0.2, inplace=True)


        self.deconv1 = nn.ConvTranspose2d(64, hidden_size[0], kernel_size=(4,4), stride=(2,2), padding=(1,1))
        self.debn1 = nn.BatchNorm2d(hidden_size[0])
        self.derelu2 = nn.ReLU(inplace=True)
        self.delrelu2 = nn.LeakyReLU(0.2, inplace=True)
        self.dedropout1 = nn.Dropout(0.2)
        self.deconv2 = nn.Conv2d(hidden_size[0], hidden_size[1], kernel_size=(1,1), stride=(1,1))
        self.debn2 = nn.BatchNorm2d(hidden_size[1])
        self.derelu3 = nn.ReLU(inplace=True)
        self.delrelu3 = nn.LeakyReLU(0.2, inplace=True)
        self.dedropout2 = nn.Dropout(0.2)
        self.deconv3 = nn.ConvTranspose2d(hidden_size[1], hidden_size[2], kernel_size=(4,4), stride=(2,2), padding=(1,1))
        self.debn3 = nn.BatchNorm2d(hidden_size[2])
        self.derelu4 = nn.ReLU(inplace=True)
        self.delrelu4 = nn.LeakyReLU(0.2, inplace=True)
        self.dedropout3 = nn.Dropout(0.2)
        self.deconv4 = nn.Conv2d(hidden_size[2], 1, kernel_size=(1,1), stride=(1,1))

    def encoder(self, x):
        self.batch_size = x.shape[0]
        x = self.lrelu1(self.bn1(self.conv1(x)))
        x = self.lrelu2(self.bn2(self.conv2(x)))
        x = self.lrelu3(self.bn3(self.conv3(x))).view(self.batch_size, -1)
        x = self.lrelu4(self.bn4(self.fc1(x)))

        return self.fc2_1(x), self.fc2_2(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(logvar / 2)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decoder(self, x, input_size = None):
        if input_size is not None:
            self.batch_size = input_size
        x = self.derelu1(self.fc3(x))

        size = int(math.sqrt(x.nelement()/self.batch_size/64))
        x = x.view(self.batch_size, 64, size,size)
        x = self.delrelu2(self.debn1(self.deconv1(x)))
        x = self.delrelu3(self.debn2(self.deconv2(x)))
        x = self.delrelu4(self.debn3(self.deconv3(x)))
        x = self.deconv4(x)
        # x = self.sigmoid(self.conv4(x))

        return x

    def forward(self, x):
        mu, log_var = self.encoder(x)
        x = self.reparameterize(mu, log_var)
        return self.decoder(x), mu, log_var

import torch
import torch.nn as nn


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class Unflatten(nn.Module):
    def __init__(self, channel, height, width):
        super(Unflatten, self).__init__()
        self.channel = channel
        self.height = height
        self.width = width

    def forward(self, input):
        return input.view(input.size(0), self.channel, self.height, self.width)


class ConvVAE(nn.Module):

    def __init__(self, latent_size):
        super(ConvVAE, self).__init__()

        self.latent_size = latent_size

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            Flatten(),
            nn.Linear(6272, 1024),
            nn.ReLU()
        )

        # hidden => mu
        self.fc1 = nn.Linear(1024, self.latent_size)

        # hidden => logvar
        self.fc2 = nn.Linear(1024, self.latent_size)

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 6272),
            nn.ReLU(),
            Unflatten(128, 7, 7),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        mu, logvar = self.fc1(h), self.fc2(h)
        return mu, logvar

    def decode(self, z):
        z = self.decoder(z)
        return z

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar