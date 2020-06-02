import os
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time
from torchvision import transforms
from torchvision.utils import save_image
from tensorboardX import SummaryWriter
from PIL import Image
from torch.utils.data import Dataset, DataLoader

import models
import load_data
import utils

if __name__ == '__main__':
    img_dir = "/media/seonghun/data1/mnist/mnist_png"
    sample_dir = "sample"
    save_model = "./AAEconv_model"
    save_result = "./result"
    tensorboard_dir = "./tensorboard/AAEconv"
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Hyper-parameters
    eps = 1e-8
    num_epochs = 500
    batch_size = 128
    input_size = 32 * 32
    input_size_sqrt = 32
    # hidden_size = [400, 20, 2]
    hidden_size = [256, 128, 49, 8]

    # train_loader = torch.utils.data.DataLoader(
    #     load_data.MNIST(img_dir, 'train'),
    #     batch_size = batch_size, shuffle = True, num_workers = 2
    # )

    a = load_data.cifar10_dataloader(imagedir="/media/seonghun/data1/cifar", mode="apple")
    train_loader = torch.utils.data.DataLoader(a, batch_size=batch_size, shuffle=True, num_workers=2)
    print(len(a))

    # model = models.ConvVAE(2)
    # model = models.AAE_conv(hidden_size, input_size)
    encoder = models.AAE_encoder(hidden_size, input_size)
    decoder = models.AAE_decoder(hidden_size, input_size)
    discriminator = models.AAE_discriminator(hidden_size, input_size)
    print(encoder)
    print(decoder)
    print(discriminator)
    encoder.load_state_dict(torch.load(os.path.join(save_model, "encoder40.pth")))
    decoder.load_state_dict(torch.load(os.path.join(save_model, "decoder40.pth")))
    discriminator.load_state_dict(torch.load(os.path.join(save_model, "discriminator40.pth")))
    encoder.apply(utils.init_weights)
    decoder.apply(utils.init_weights)
    discriminator.apply(utils.init_weights)

    en_optimizer = optim.Adam(encoder.parameters(), lr=3e-4)
    de_optimizer = optim.Adam(decoder.parameters(), lr=3e-4)
    dis_optimizer = optim.Adam(discriminator.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    if os.path.exists(save_result) == False:
        os.mkdir(save_result)
    if os.path.exists(save_model) == False:
        os.mkdir(save_model)
    if os.path.exists(tensorboard_dir) == False:
        os.mkdir(tensorboard_dir)
    i = 0
    while (True):
        if os.path.exists(os.path.join(tensorboard_dir, str(i))) == True:
            i += 1
            continue
        os.mkdir(os.path.join(tensorboard_dir, str(i)))
        print("tensorboard = {}".format(i))
        break
    # Start training
    j = 0
    for epoch in range(1):
        loss_recon = 0
        loss_dis = 0
        loss_gen = 0
        dis_real_acc = 0
        dis_fake_acc = 0

        stime = time.time()

        for i, (org_image, _) in enumerate(train_loader):
            image = org_image + torch.randn(org_image.size())
            image = image.to(device)
            # org_image = org_image.to(device)
            org_image = org_image.to(device)  # for CAAE
            # org_image = org_image.to(device).view(-1, input_size)     # for FCAAE
            encoder = encoder.to(device)
            decoder = decoder.to(device)
            discriminator = discriminator.to(device)
            encoder.eval()
            decoder.eval()
            discriminator.eval()

        with torch.no_grad():
            decoder.eval()
            encoder.eval()
            utils.get_sample(encoder, decoder, hidden_size, org_image, sample_dir, epoch, batch_size, device, input_size)

