import os
where = "local"
if where == "server2":
    img_dir = "/home/seonghun20/data/MVTec"
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
elif where == "server1":
    img_dir = "/home/seonghun/anomaly/data/MVTec"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
elif where == "local":
    img_dir = "/media/seonghun/data1/MVTec"

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

import models_VAE as models
import load_data
import utils
import testing


def l1loss(pred, target):
    loss = torch.abs(pred - target)
    return loss.mean()

def FMloss(pred, target):
    loss = torch.mean((pred-target)**2)/2
    return loss



if __name__ == '__main__':
    load = False
    check_epoch = 0

    img_fol_name = "cable"
    mode = "train_one"
    sample_dir = "./MVTec_sample/sample/"
    save_model = "./VAE_MVTec_model"
    tensorboard_dir = "./tensorboard/VAE_MVTec"
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    init_lr = 5e-4
    warm_up_epoch = 5
    window_size = 9
    kld_warm_up = 200
    FM_warm_up = 200

    FM_epoch = -1


    # Hyper-parameters
    eps = 1e-8
    num_epochs = 300
    batch_size = 16
    input_size = 256*256
    input_size_sqrt = 256
    beta1 = 0.5
    hidden_size = [64, 128, 256, 512, 800]        # last -> latent dimension // increase complexity, increase latent dimension
    fol = ""
    for i in hidden_size:
        fol+=str(i)


    ### data config
    normal_data = load_data.MVTec_dataloader(image_dir=os.path.join(img_dir,img_fol_name), mode=mode, in_size = input_size_sqrt)
    train_loader = torch.utils.data.DataLoader(normal_data, batch_size=batch_size, shuffle=True, num_workers=2)

    abnormal_data = load_data.MVTec_dataloader(image_dir=os.path.join(img_dir,img_fol_name), mode="test", in_size = input_size_sqrt)
    anomaly_loader = torch.utils.data.DataLoader(abnormal_data, batch_size=8, shuffle=True, num_workers=2)

    test_normal_loader = torch.utils.data.DataLoader(load_data.MVTec_dataloader(image_dir = os.path.join(img_dir, img_fol_name), mode="test_normal", in_size = input_size_sqrt)
                                                   , batch_size=batch_size, shuffle=True, num_workers=2)


    ##### model, optimizer config
    model = models.VAE(hidden_size, input_size)
    print(model)
    FM_model = torchvision.models.resnet50(pretrained=True)
    feature1 = utils.GradCAM(FM_model, "layer1")
    feature2 = utils.GradCAM(FM_model, "layer2")
    feature3 = utils.GradCAM(FM_model, "layer3")
    feature4 = utils.GradCAM(FM_model, "layer4")

    # optimizer = optim.Adam(model.parameters(), lr=2e-4, weight_decay=1e-5, betas=(beta1, 0.999))
    optimizer = optim.Adam(model.parameters(), lr=init_lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [80, 120, 160, 200], gamma = 0.2)


    if load == True:
        print("load checkpoint {}".format(check_epoch))

        checkpoint = torch.load(os.path.join(save_model, fol, "checkpoint{}.pth.tar".format(check_epoch)))
        if hidden_size != checkpoint["hidden_size"]:
            raise AttributeError("checkpoint's hidden size is not same")


        ##### load model
        model.load_state_dict(checkpoint["model"])

        # optimizer.load_state_dict(checkpoint["optimizer"])
        # utils.opt_cuda_setting(optimizer)


    else:
        model.apply(utils.init_weights)



    #### loss config
    criterion = nn.BCEWithLogitsLoss(reduction="mean")
    MSE = nn.MSELoss(reduction="mean")
    SSIM = utils.SSIM(window_size)
    # criterion = nn.BCEWithLogitsLoss(reduction="mean")




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
    score = 0
    optimizer.step()
    for epoch in range(num_epochs):
        PM=0
        FM=0
        KLD=0
        ssim = 0
        stime = time.time()
        # kld_weight = kld_warm_up
        if epoch < kld_warm_up:
            kld_weight = (kld_warm_up / (epoch + 1))
        else: kld_weight = 1
        print("kld = {}".format(kld_weight))

        if epoch < FM_warm_up:
            FM_weight = ((epoch + 1) / (FM_warm_up))
        else: FM_weight = 1
        print("FM = {}".format(FM_weight))
        for i, (org_image, _) in enumerate(train_loader):

            org_image = org_image.to(device)
            FM_model = FM_model.to(device).eval()
            for param in FM_model.parameters():
                param.requires_grad = False

            model = model.to(device).train()
            optimizer.zero_grad()

            recon, mu, logvar = model(org_image)

            PM_loss = MSE(recon, org_image)
            # PM_loss = l1loss(recon, org_image)
            SSIM_loss = 1 - SSIM(recon, org_image)
            KLD_loss = (kld_weight) * - 0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp()).mean()

            if epoch>FM_epoch:
                FM_model(org_image)
                FM_1 = feature1.activation
                FM_2 = feature2.activation
                FM_3 = feature3.activation
                FM_4 = feature4.activation
                FM_model(recon)
                FM2_1 = feature1.activation
                FM2_2 = feature2.activation
                FM2_3 = feature3.activation
                FM2_4 = feature4.activation

                FM_loss = FM_warm_up*(FMloss(FM_1, FM2_1) + FMloss(FM_2, FM2_2) + FMloss(FM_3, FM2_3) + FMloss(FM_4, FM2_4))
                # loss = PM_loss + KLD_loss + FM_loss
                loss = PM_loss + KLD_loss + SSIM_loss + FM_loss
                FM += FM_loss.item()
            else:
                loss = PM_loss + KLD_loss + SSIM_loss
            # loss = PM_loss + KLD_loss + SSIM_loss
            loss.backward()
            optimizer.step()
            PM += PM_loss.item()
            KLD += KLD_loss.item()
            ssim += SSIM_loss.item()

        print('Epoch [{}/{}], Step {}, exe time: {:.2f}, lr: {:.4f}*e-4'
                  .format(epoch, num_epochs, i+1, time.time() - stime, scheduler.get_lr()[0] * 10** 4))

        print('PM_loss: {:.4f}, FM_loss: {:.4f}, KLD_loss: {:.4f}, SSIM_loss : {:.4f}, Total_loss {:.4f}'
              .format(PM/i, FM/i, KLD/i, ssim/i, (ssim+PM+KLD)/i))
        summary.add_scalar('loss/PM_loss', PM/i, j)
        summary.add_scalar('loss/FM_loss', FM/i, j)
        summary.add_scalar('loss/KLD_loss', KLD/i, j)
        summary.add_scalar('loss/SSIM_loss', ssim/i, j)
        summary.add_scalar('loss/loss', (ssim+PM+KLD)/i, j)
        summary.add_scalar("learning_rate/lr", (scheduler.get_lr()[0]), j)
        time.sleep(0.001)
        torch.save({
            'hidden_size' : hidden_size,
            'model': model.state_dict(),
            }, os.path.join(save_model, fol, 'checkpoint_last.pth.tar'))
        scheduler.step(epoch)
        stime = time.time()
        with torch.no_grad():
            model.eval()
            FM_model.eval()
            utils.VAE_get_sample(model, hidden_size, org_image, os.path.join(sample_dir, fol), epoch, 4, device, input_size)
            score, abs_score = testing.testing_VAE(model, test_normal_loader, anomaly_loader, hidden_size, j, device, summary)
            print("score : {:.4f}, abs_score : {:.4f}, eval time : {:.4f}".format(score, abs_score, time.time() - stime))
            summary.add_scalar("score/AUC", score, j)
            summary.add_scalar("score/AUC_abs", abs_score, j)
            if best_score < abs_score:
                best_score = abs_score
                torch.save({
                    'hidden_size' : hidden_size,
                    'model': model.state_dict(),
                    }, os.path.join(save_model, fol, 'checkpoint_best.pth.tar'))
                print("save best score")

            if epoch % 5 == 4:
                testing.testing(model, anomaly_loader, os.path.join(sample_dir, fol), hidden_size, input_size, epoch, device)
        j += 1
