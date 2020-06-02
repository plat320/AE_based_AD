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
from torch.autograd import Variable

import models_AAE as models
import load_data
import utils
import testing


def l1loss(pred, target):
    return torch.mean(torch.abs(pred - target))

if __name__ == '__main__':
    where = "local"
    load = False
    check_epoch = 0
    if where == "server2":
        img_dir = "/home/seonghun20/data/MVTec"
        os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    elif where == "server1":
        img_dir = "/home/seonghun/anomaly/data/MVTec"
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    elif where == "local":
        img_dir = "/media/seonghun/data1/MVTec"

    init_lr = 5e-4
    dis_lr_ratio = 1
    warm_up_epoch = 20
    window_size = 9
    kld_warm_up = 100
    FM_warm_up = 200
    FM_epoch = 40

    # Hyper-parameters
    eps = 1e-8
    num_epochs = 300
    batch_size = 16
    input_size = 256*256
    input_size_sqrt = 256
    hidden_size = [64, 128, 256, 512, 800]        # last -> latent dimension // increase complexity, increase latent dimension
    beta1 = 0.5


    fol = ""
    for i in hidden_size:
        fol+=str(i)
    img_fol_name = "cable"
    mode = "train_one"
    sample_dir = "./MVTec_sample/sample_cable/"
    save_model = "./AAE_MVTec_model"
    tensorboard_dir = "./tensorboard/AAE_MVTec"
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    ##### model, optimizer config
    encoder = models.AAE_encoder(hidden_size, input_size)
    decoder = models.AAE_decoder(hidden_size, input_size)
    discriminator = models.AAE_discriminator(hidden_size, input_size)


    # FM_model = torchvision.models.resnet50(pretrained=True)
    # feature1 = utils.GradCAM(FM_model, "layer1")
    # feature2 = utils.GradCAM(FM_model, "layer2")
    # feature3 = utils.GradCAM(FM_model, "layer3")
    # feature4 = utils.GradCAM(FM_model, "layer4")

    en_optimizer = optim.Adam(encoder.parameters(), lr = init_lr, weight_decay=1e-5)
    de_optimizer = optim.Adam(decoder.parameters(), lr = init_lr, weight_decay=1e-5)
    dis_optimizer = optim.Adam(discriminator.parameters(), lr = init_lr/dis_lr_ratio, weight_decay=1e-5)

    if load == True:
        print("load checkpoint {}".format(check_epoch))

        checkpoint = torch.load(os.path.join(save_model, fol, "checkpoint{}.pth.tar".format(check_epoch)))
        if hidden_size != checkpoint["hidden_size"]:
            raise AttributeError("checkpoint's hidden size is not same")

        ##### load model
        encoder.load_state_dict(checkpoint['encoder'])
        decoder.load_state_dict(checkpoint['decoder'])
        discriminator.load_state_dict(checkpoint['discriminator'])

        ##### load optimizer
        # en_optimizer.load_state_dict(checkpoint['en_optimizer'])
        # utils.opt_cuda_setting(en_optimizer)
        # de_optimizer.load_state_dict(checkpoint['de_optimizer'])
        # utils.opt_cuda_setting(de_optimizer)
        # dis_optimizer.load_state_dict(checkpoint['dis_optimizer'])
        # utils.opt_cuda_setting(dis_optimizer)

    else:
        encoder.apply(utils.init_weights)
        decoder.apply(utils.init_weights)
        discriminator.apply(utils.init_weights)


    ### data config
    normal_data = load_data.MVTec_dataloader(image_dir=os.path.join(img_dir,img_fol_name), mode=mode, in_size = input_size_sqrt)
    train_loader = torch.utils.data.DataLoader(normal_data, batch_size=batch_size, shuffle=True, num_workers=2)

    abnormal_data = load_data.MVTec_dataloader(image_dir=os.path.join(img_dir,img_fol_name), mode="test", in_size = input_size_sqrt)
    anomaly_loader = torch.utils.data.DataLoader(abnormal_data, batch_size=8, shuffle=True, num_workers=2)

    test_normal_loader = torch.utils.data.DataLoader(load_data.MVTec_dataloader(image_dir = os.path.join(img_dir, img_fol_name), mode="test_normal", in_size = input_size_sqrt)
                                                   , batch_size=batch_size, shuffle=True, num_workers=2)


    BCE_real = nn.BCEWithLogitsLoss()
    BCE_recon = nn.BCEWithLogitsLoss(reduction="mean")
    BCE_fake = nn.BCEWithLogitsLoss()
    BCE_gen = nn.BCEWithLogitsLoss()
    SSIM_criterion = utils.SSIM(window_size)
    MSE_criterion = nn.MSELoss(reduction="mean")
    MSE = nn.MSELoss(reduction="mean")

    if os.path.exists(os.path.join(save_model, fol)) == False:
        os.mkdir(os.path.join(save_model, fol))
    if os.path.exists(tensorboard_dir) == False:
        os.mkdir(tensorboard_dir)
    if os.path.exists(os.path.join(sample_dir, fol)) == False:
        os.mkdir(os.path.join(sample_dir, fol))
    if os.path.exists(os.path.join(tensorboard_dir, fol)) == False:
        os.mkdir(os.path.join(tensorboard_dir, fol))
    i=0
    while(True):
        if os.path.exists(os.path.join(tensorboard_dir, fol, str(i))) == True:
            i+=1
            continue
        os.mkdir(os.path.join(tensorboard_dir, fol, str(i)))
        break
    print("tensorboard = {}".format(fol))
    summary = SummaryWriter(os.path.join(tensorboard_dir, fol, str(i)))
    # Start training
    j=0
    best_score=0
    for epoch in range(num_epochs):
        loss_recon=0
        loss_SSIM=0
        loss_BCE=0
        loss_dis=0
        loss_gen=0
        FM= 0

        dis_real_acc = 0
        dis_fake_acc = 0


        stime = time.time()
        for i, (org_image, _) in enumerate(train_loader):
            org_image = org_image.to(device)                            # for CAAE
            encoder = encoder.to(device)
            decoder = decoder.to(device)
            discriminator = discriminator.to(device)
            encoder.train()
            decoder.train()
            discriminator.train()

            en_optimizer.zero_grad()
            de_optimizer.zero_grad()
            dis_optimizer.zero_grad()

            ############### train auto-encoder
            latent_vector = encoder(org_image)
            recon = decoder(latent_vector)
            PM_loss = MSE(recon, org_image)

            # if epoch>FM_epoch:
            #     FM_model(org_image)
            #     FM_1 = feature1.activation
            #     FM_2 = feature2.activation
            #     FM_3 = feature3.activation
            #     FM_4 = feature4.activation
            #     FM_model(recon)
            #     FM2_1 = feature1.activation
            #     FM2_2 = feature2.activation
            #     FM2_3 = feature3.activation
            #     FM2_4 = feature4.activation
            #
            #     FM_loss = FM_warm_up*(FMloss(FM_1, FM2_1) + FMloss(FM_2, FM2_2) + FMloss(FM_3, FM2_3) + FMloss(FM_4, FM2_4))
            #     loss = PM_loss + KLD_loss + FM_loss
            #     loss = PM_loss + FM_loss
            #     FM += FM_loss.item()
            # else:
            #     loss = PM_loss

            # SSIM_loss = SSIM_criterion(recon, org_image)
            # loss = (SSIM_loss+recon_loss)
            PM_loss.backward()
            en_optimizer.step()
            de_optimizer.step()
            # en_optimizer.zero_grad()
            # de_optimizer.zero_grad()                # for using updated auto encoder

            ############ train discriminator
            dis_optimizer.zero_grad()

            z_real = torch.randn(org_image.shape[0], hidden_size[-1], 1, 1).to(device)
            D_real = discriminator(z_real)

            dis_real_acc += torch.sum(torch.sigmoid(D_real).cpu().detach()>0.5).item()

            D_real_loss = BCE_real(D_real, torch.ones_like(D_real))
            D_real_loss.backward()

            z_fake = encoder(org_image)
            D_fake = discriminator(z_fake.detach())

            dis_fake_acc += torch.sum(torch.sigmoid(D_fake).cpu().detach()<0.5).item()

            D_fake_loss = BCE_fake(D_fake, torch.zeros_like(D_fake))
            D_fake_loss.backward()

            D_loss = D_real_loss + D_fake_loss

            dis_optimizer.step()
            # D_loss = (-torch.mean(torch.log(D_real + eps) + torch.log(1 - D_fake + eps)))

            ########### train generator(encoder)
            en_optimizer.zero_grad()
            z_fake = encoder(org_image)
            D_fake = discriminator(z_fake)
            G_loss = BCE_gen(D_fake, torch.ones_like(D_fake))

            G_loss.backward()
            en_optimizer.step()

            loss_recon += PM_loss.item()
            # loss_recon += recon_loss.item()
            loss_dis += D_loss.item()
            loss_gen += G_loss.item()
            # loss_SSIM += SSIM_loss.item()



        torch.save({
            'hidden_size' : hidden_size,
            'encoder': encoder.state_dict(),
            'decoder': decoder.state_dict(),
            'discriminator': discriminator.state_dict(),
            }, os.path.join(save_model, fol, 'checkpoint_last.pth.tar'))



        print('Epoch [{}/{}], Step {}, exe time: {:.2f} '
              .format(epoch, num_epochs, i+1, time.time() - stime, ) )
        print('recon_loss: {:.4f}, FM_loss: {:.4f}, dis_loss: {:.4f}, gen_loss: {:.4f},  Total_loss {:.4f}'
              .format(loss_recon/i, FM/i, loss_dis/i, loss_gen/i, (loss_dis + loss_gen + loss_recon)/i))
        print("Discriminator acc --- Real : {:.4f}, Fake : {:.4f}".format(dis_real_acc / len(normal_data), dis_fake_acc / len(normal_data)))
        summary.add_scalar('loss/recon_loss', loss_recon/i, j)
        summary.add_scalar('loss/dis_loss', loss_dis/i, j)
        summary.add_scalar('loss/gen_loss', loss_gen/i, j)
        summary.add_scalar('loss/total_loss', (loss_recon+loss_gen+loss_dis)/i, j)
        j += 1
        time.sleep(0.001)
        with torch.no_grad():
            decoder.eval()
            encoder.eval()
            discriminator.eval()
            utils.AAE_get_sample(encoder,decoder, hidden_size, org_image, os.path.join(sample_dir, fol), epoch, batch_size, device, input_size)
            score, abs_score = testing.testing_AAE(encoder, test_normal_loader, anomaly_loader, hidden_size, j, device, summary)
            stime = time.time()
            print("score : {:.4f}, abs_score : {:.4f}, eval time : {:.4f}".format(score, abs_score, time.time() - stime))
            summary.add_scalar("score/AUC", score, j)
            summary.add_scalar("score/AUC_abs", abs_score, j)
            if abs_score > best_score:
                print("save best score checkpoint")
                torch.save({
                    'hidden_size' : hidden_size,
                    'encoder': encoder.state_dict(),
                    'decoder': decoder.state_dict(),
                    'discriminator': discriminator.state_dict(),
                    }, os.path.join(save_model, fol, 'checkpoint_best.pth.tar'))
                best_score = abs_score
            if epoch % 5 == 0:
                testing.testing(encoder, decoder, anomaly_loader, os.path.join(sample_dir, fol), hidden_size, input_size, epoch, device)
