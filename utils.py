import os
import time
import math
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from torch.optim.lr_scheduler import _LRScheduler
from torchvision import transforms
from torchvision.utils import save_image
from tensorboardX import SummaryWriter
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from math import exp

import load_data
import utils


class WarmUpLR(_LRScheduler):

    def __init__(self, optimizer, warmup_step, start_lr, end_lr, last_epoch=-1):

        self.optimizer = optimizer
        self.warmup_step = warmup_step

        self.slope    = (end_lr - start_lr) / (warmup_step + 1e-8)
        self.start_lr = start_lr

        super().__init__(optimizer, last_epoch)



    def get_lr(self):
        # last_epoch is not 'epoch'. It is used as 'step' in this code.
        step = self.last_epoch
        return [self.slope * step + self.start_lr]

def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    return x


def init_weights(m):
    if type(m) == nn.Linear:
        '''
        He initializer
        '''
        torch.nn.init.kaiming_normal_(m.weight)
        m.bias.data.fill_(0.01)

def plt_manifold(save_model, model, device, save_file_path, input_size, epoch, mean_range=3, n=20, figsize=(8, 10)): #
    model.load_state_dict(torch.load(os.path.join(save_model, "model%d.pth"%(epoch))))
    input_size = int(math.sqrt(input_size))
    x_axis = np.linspace(-mean_range, mean_range, n)
    y_axis = np.linspace(-mean_range, mean_range, n)
    canvas = np.empty((input_size*n, input_size*n))

    for i, yi in enumerate(x_axis):
        for j, xi in enumerate(y_axis):
            z_mean = np.array([[xi, yi]] * 1)
            z_mean = torch.tensor(z_mean, device=device).float()
            x_reconst = model.Decoder(z_mean)
            # x_reconst = torch.sigmoid(model.decoder(z_mean, 1))
            x_reconst = x_reconst.detach().cpu().numpy()
            canvas[(n-i-1)*input_size:(n-i)*input_size, j*input_size:(j+1)*input_size] = x_reconst[0].reshape(3,input_size, input_size) # for 3channel

    plt.figure(figsize=figsize)
    xi, yi = np.meshgrid(x_axis, y_axis)
    plt.imshow(canvas, origin="upper")
    plt.savefig(os.path.join(save_file_path, "{}.png".format(epoch)))
    # plt.show()


def opt_cuda_setting(opti):
        for state in opti.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()



def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / (gauss.sum())
    # return gauss / (gauss.sum() + 1e-8)


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    # C1 = 0.01 ** 2
    # C2 = 0.03 ** 2
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2) + 1e-8)

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(torch.nn.Module):
    def __init__(self, window_size=15, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)



class GradCAM(nn.Module):
    def __init__(self, model, grad_layer='reduction'):
        super().__init__()
        self.model = model


        self.activation = None
        self.gradient = None
        self._register_hooks(grad_layer)

    def _register_hooks(self, grad_layer):
        def forward_hook(module, input, output):
            self.activation = output.clone()
            # self.activation = output.detach().clone()

        def backward_hook(module, grad_input, grad_output):
            self.gradient = grad_output[0].detach().clone()

        gradient_layer_found = False
        for name, module in self.model.named_modules():
            if name == grad_layer:
                module.register_forward_hook(forward_hook)
                module.register_backward_hook(backward_hook)
                gradient_layer_found = True
                break

        if gradient_layer_found is False:
            raise AttributeError('Gradient layer %s not found in the internal model' % grad_layer)

        return self.activation

def get_sample_cifar(encoder, decoder, hidden_size, images, sample_dir, epoch, batch_size, device, input_size):
    input_size = int(math.sqrt(input_size))
    # z = torch.randn(batch_size, hidden_size[-1], int(input_size/8), int(input_size/8)).to(device)  # Randomly Sample z (Only Contains Mean)
    # out = decoder(z).view(-1, 3, input_size, input_size)
    # out = to_img(out)
    # save_image(out, os.path.join(sample_dir, 'sampled-{}.png'.format(epoch + 1)))

    # Save the reconstructed images
    # out = decoder(encoder(images))
    out = (torch.sigmoid(torch.cat(decoder(encoder(images)), dim = 1)))
    # out = (torch.sigmoid(torch.cat(decoder(encoder(images)), dim = 1))-0.5)*2
    x_concat = torch.cat([(images.view(-1, 3, input_size, input_size)), out.view(-1, 3, input_size, input_size)], dim=2)
    save_image(x_concat, os.path.join(sample_dir, 'reconst-{}.png'.format(epoch)))


def AAE_get_sample(encoder, decoder, hidden_size, images, sample_dir, epoch, batch_size, device, input_size):
    input_size = int(math.sqrt(input_size))

    # Save the reconstructed images
    out = decoder(encoder(images))
    out=torch.tanh(out)

    x_concat = torch.cat([(images.view(-1, 3, input_size, input_size)/2)+0.5, (out.view(-1, 3, input_size, input_size)/2)+0.5], dim=2)

    # x_concat = F.max_pool2d(x_concat, (2,2))
    save_image(x_concat, os.path.join(sample_dir, 'reconst-{}.png'.format(epoch)))

    # Save the sampled image
    # z = torch.randn(batch_size, 1, int(input_size/128), int(input_size/128)).to(device)  # Randomly Sample z (Only Contains Mean)
    # out = model.decoder(z).view(-1, 3, input_size, input_size)
    # out = to_img(out)
    # save_image(out, os.path.join(sample_dir, 'sampled-{}.png'.format(epoch + 1)))


def get_sample(encoder, decoder, hidden_size, images, sample_dir, epoch, batch_size, device, input_size):
    input_size = int(math.sqrt(input_size))
    # z = torch.randn(batch_size, hidden_size[-1], int(input_size/8), int(input_size/8)).to(device)  # Randomly Sample z (Only Contains Mean)
    # out = decoder(z).view(-1, 3, input_size, input_size)
    # out = to_img(out)
    # save_image(out, os.path.join(sample_dir, 'sampled-{}.png'.format(epoch + 1)))

    # Save the reconstructed images
    # out = decoder(encoder(images))
    out = (torch.sigmoid(torch.cat(decoder(encoder(images)), dim = 1)))
    # out = (torch.sigmoid(torch.cat(decoder(encoder(images)), dim = 1))-0.5)*2
    x_concat = torch.cat([(images.view(-1, 3, input_size, input_size)+0.5)/2, out.view(-1, 3, input_size, input_size)], dim=2)
    save_image(x_concat, os.path.join(sample_dir, 'reconst-{}.png'.format(epoch)))


def get_sample_for_fc(model, hidden_size, images, sample_dir, epoch, batch_size, device, input_size):
    input_size = int(math.sqrt(input_size))
    # z = torch.randn(batch_size, hidden_size[-1], int(input_size/8), int(input_size/8)).to(device)  # Randomly Sample z (Only Contains Mean)
    # out = decoder(z).view(-1, 3, input_size, input_size)
    # out = to_img(out)
    # save_image(out, os.path.join(sample_dir, 'sampled-{}.png'.format(epoch + 1)))

    # Save the reconstructed images
    _, out, _ = model(images)
    x_concat = torch.cat([(images.view(-1, 3, input_size, input_size)), out.view(-1, 3, input_size, input_size)], dim=2)
    save_image(x_concat, os.path.join(sample_dir, 'reconst-{}.png'.format(epoch)))

if __name__ == '__main__':
    input_size = 512*512
    hidden_size = [256, 128, 64, 32]  # decrease channel
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = models.AAE_encoder(hidden_size, input_size)
    decoder = models.AAE_decoder_split(hidden_size, input_size)
    discriminator = models.AAE_discriminator(hidden_size, input_size)

    encoder.load_state_dict(torch.load(os.path.join("./AAE_MVTec_model", "encoder199.pth")))
    decoder.load_state_dict(torch.load(os.path.join("./AAE_MVTec_model", "decoder199.pth")))
    discriminator.load_state_dict(torch.load(os.path.join("./AAE_MVTec_model", "discriminator199.pth")))

    a = load_data.MVTec_dataloader(image_dir=os.path.join("/media/seonghun/data1/MVTec", "cable"), mode="train_one")
    train_loader = torch.utils.data.DataLoader(a, batch_size=16, shuffle=True, num_workers=2)

    for i, (org_image, _) in enumerate(train_loader):
        org_image = org_image.to(device)
        encoder = encoder.to(device).eval()
        decoder = decoder.to(device).eval()
        discriminator = discriminator.to(device).eval()

        sample_dir = "./MVTec_sample/test"

        get_sample(encoder,decoder, hidden_size, org_image, sample_dir, i+300, 16, device, 512*512)


