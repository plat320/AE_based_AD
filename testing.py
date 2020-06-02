import os
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import cv2
import statistics
from torchvision import transforms
from torchvision.utils import save_image
from tensorboardX import SummaryWriter
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, plot_roc_curve, roc_curve, auc
import time
import math

import models_VAE as models
import load_data
import utils


def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return eps.mul(std).add_(mu)

def testing(model, loader, sample_dir, hidden_size, input_size, epoch, device):
    for (org_image, _) in loader:
        org_image = org_image.to(device)
        model = model.to(device).eval()

        utils.VAE_get_sample(model, hidden_size, org_image, sample_dir, "abnormal{}".format(epoch), 4, device, input_size)
        break

def testing_VAE(model, normal_loader, abnormal_loader, hidden_size, epoch, device, summary):
    batch_size = 8

    fol = ""
    for i in hidden_size:
        fol+=str(i)

    normal = []
    normal_abs = []
    abnormal = []
    abnormal_abs = []
    with torch.no_grad():
        for (org_image, gt) in normal_loader:
            org_image = org_image.to(device)
            model.eval()

            mu, logvar  = model.encoder(org_image)
            A = torch.abs(torch.sum(mu, dim = (1,2,3))) + torch.abs(torch.sum(logvar, dim = (1,2,3)))
            A_abs = torch.sum(torch.abs(mu), dim = (1,2,3)) + torch.sum(torch.abs(logvar), dim = (1,2,3))
            # print(torch.mean(mu, dim = (1,2,3)))
            # print(torch.mean(logvar, dim = (1,2,3)))
            normal.extend(A.cpu().detach().numpy())
            normal_abs.extend(A_abs.cpu().detach().numpy())

    for (org_image, gt) in abnormal_loader:
            org_image = org_image.to(device)
            model.eval()

            mu, logvar  = model.encoder(org_image)
            B = torch.abs(torch.sum(mu, dim = (1,2,3))) + torch.abs(torch.sum(logvar, dim = (1,2,3)))
            B_abs = torch.sum(torch.abs(mu), dim = (1,2,3)) + torch.sum(torch.abs(logvar), dim = (1,2,3))
            # print(torch.mean(mu, dim = (1,2,3)))
            # print(torch.mean(logvar, dim = (1,2,3)))
            abnormal.extend(B.cpu().detach().numpy())
            abnormal_abs.extend(B_abs.cpu().detach().numpy())


    normal_label = np.zeros_like(normal)
    abnormal_label = np.ones_like(abnormal)
    print("normal mean : {:.4f}, abnormal mean : {:.4f}, normal abs : {:.4f}, abnormal abs : {:.4f}".format(statistics.mean(normal), statistics.mean(abnormal), statistics.mean(normal_abs), statistics.mean(abnormal_abs)))
    summary.add_scalar("score/normal_abs", statistics.mean(normal_abs), epoch)
    summary.add_scalar("score/abnormal_abs", statistics.mean(abnormal_abs), epoch)
    score = roc_auc_score(np.hstack((normal_label, abnormal_label)), np.hstack((normal, abnormal)))
    abs_score = roc_auc_score(np.hstack((normal_label, abnormal_label)), np.hstack((normal_abs, abnormal_abs)))
    return score, abs_score



def VAE_get_sample(model, hidden_size, images, sample_dir, epoch, batch_size, device, input_size):
    input_size = int(math.sqrt(input_size))

    # Save the reconstructed images
    out, _, _ = model(images)
    out=torch.tanh(out)

    # x_concat = torch.cat([(images.view(-1, 3, input_size, input_size)/2)+0.5, (out.view(-1, 3, input_size, input_size)/2)+0.5], dim=2)

    # x_concat = F.max_pool2d(x_concat, (2,2))
    # save_image(x_concat, os.path.join(sample_dir, 'reconst-{}.png'.format(epoch)))

    # Save the sampled image
    z = torch.randn(batch_size, 1, 1, 1).to(device)  # Randomly Sample z (Only Contains Mean)
    out = model.decoder(z).view(-1, 3, input_size, input_size)
    out = to_img(out)
    save_image(out, os.path.join(sample_dir, 'sampled-{}.png'.format(epoch + 1)))


def plt_manifold(save_model, model, device, save_file_path, input_size, epoch, mean_range=3, n=5, figsize=(8, 10)): #
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
    # plt.savefig(os.path.join(save_file_path, "{}.png".format(epoch)))
    plt.show()

def vec2recon(model, hidden_size, z_mean = None):
    model.eval()
    # z_mean = torch.ones((1, hidden_size[-1], 1, 1))*0.01
    if z_mean == None:
        for i in range(100):
            z_mean = torch.ones((1,hidden_size[-1], 1,1))*i
            # z_mean[:, i,:,:] = 0.1
            print(z_mean.view(hidden_size[-1]))
            z_mean = torch.tensor(z_mean, device=device).float()
            x_reconst = torch.sigmoid(model.decoder(z_mean))
            x_reconst = x_reconst.squeeze().detach().cpu().numpy().transpose(1, 2, 0)
            plt.imshow(x_reconst)
            plt.show()
    else:
        print(z_mean.view(hidden_size[-1]))
        z_mean = torch.tensor(z_mean, device=device).float()
        x_reconst = torch.sigmoid(model.decoder(z_mean))
        x_reconst = x_reconst.squeeze().detach().cpu().numpy().transpose(1, 2, 0)
        plt.imshow(x_reconst)
        plt.show()



if __name__ == '__main__':
    img_dir = "/media/seonghun/data1/MVTec"
    sample_dir = "sample"
    save_model = "./VAE_MVTec_model"
    save_result = "./result"
    tensorboard_dir = "./tensorboard/AAE"
    img_fol_name = "cable"
    mode = "train_one"
    input_size = 256*256
    input_size = int(math.sqrt(input_size))
    batch_size = 8
    hidden_size = [64, 128, 256, 512, 800]
    save_model = "./VAE_MVTec_model"

    check_epoch = "_last"

    fol = ""
    for i in hidden_size:
        fol+=str(i)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    normal_data = load_data.MVTec_dataloader(image_dir=os.path.join(img_dir,img_fol_name), mode=mode, in_size=input_size)
    train_loader = torch.utils.data.DataLoader(normal_data, batch_size=batch_size, shuffle=True, num_workers=2)
    abnormal_data = load_data.MVTec_dataloader(image_dir=os.path.join(img_dir,img_fol_name), mode="test", in_size=input_size)
    anomaly_loader = torch.utils.data.DataLoader(abnormal_data, batch_size=batch_size, shuffle=True, num_workers=2)
    test_normal_loader = torch.utils.data.DataLoader(load_data.MVTec_dataloader(image_dir = os.path.join(img_dir, img_fol_name), mode="test_normal", in_size = input_size)
                                                   , batch_size=batch_size, shuffle=True, num_workers=2)

    model = models.VAE(hidden_size, input_size)
    model = model.to(device)
    checkpoint = torch.load(os.path.join(save_model, "test", "checkpoint_best.pth.tar"))
    model.load_state_dict(checkpoint["model"])

    print("Test normal")
    normal = []
    abnormal = []
    normal_abs = []
    abnormal_abs = []
    with torch.no_grad():
        for (org_image, gt) in test_normal_loader:
            org_image = org_image.to(device)
            model.eval()

            mu, logvar  = model.encoder(org_image)
            A = torch.abs(torch.sum(mu, dim = (1,2,3))) + torch.abs(torch.sum(logvar, dim = (1,2,3)))
            A_abs = torch.sum(torch.abs(mu), dim = (1,2,3)) + torch.sum(torch.abs(logvar), dim = (1,2,3))
            # print(torch.mean(mu, dim = (1,2,3)))
            # print(torch.mean(logvar, dim = (1,2,3)))
            normal.extend(A.cpu().detach().numpy())
            normal_abs.extend(A_abs.cpu().detach().numpy())
        print(normal_abs)

        for (org_image, gt) in anomaly_loader:
            org_image = org_image.to(device)
            model.eval()

            mu, logvar  = model.encoder(org_image)
            B = torch.abs(torch.sum(mu, dim = (1,2,3))) + torch.abs(torch.sum(logvar, dim = (1,2,3)))
            B_abs = torch.sum(torch.abs(mu), dim = (1,2,3)) + torch.sum(torch.abs(logvar), dim = (1,2,3))
            # print(torch.mean(mu, dim = (1,2,3)))
            # print(torch.mean(logvar, dim = (1,2,3)))
            abnormal.extend(B.cpu().detach().numpy())
            abnormal_abs.extend(B_abs.cpu().detach().numpy())
    print(abnormal_abs)

    normal_label = np.zeros_like(normal)
    abnormal_label = np.ones_like(abnormal)
    print("normal mean : {:.4f}, abnormal mean : {:.4f}, normal abs : {:.4f}, abnormal abs : {:.4f}".format(statistics.mean(normal), statistics.mean(abnormal), statistics.mean(normal_abs), statistics.mean(abnormal_abs)))
    score = roc_auc_score(np.hstack((normal_label, abnormal_label)), np.hstack((normal, abnormal)))
    abs_score = roc_auc_score(np.hstack((normal_label, abnormal_label)), np.hstack((normal_abs, abnormal_abs)))

    result = dict()
    grid_num = 40
    a = np.zeros(grid_num)
    TP = np.zeros(grid_num)
    FN = np.zeros(grid_num)
    TN = np.zeros(grid_num)
    FP = np.zeros(grid_num)
    TPR = np.zeros(grid_num)
    FPR = np.zeros(grid_num)
    save_num = 0

    # print("discrete accuracy")
    # for num in range(grid_num):
    #     b = num / grid_num
    #     TP[num] = np.sum(normal <= b) / normal.size
    #     TN[num] = np.sum(abnormal <= b) / abnormal.size
    #     FN[num] = np.sum(normal > b) / normal.size
    #     FP[num] = np.sum(abnormal > b) / abnormal.size
    #     a[num] = b
    #     TPR[num] = TP[num] / (TP[num] + FN[num])
    #     FPR[num] = 1 - FP[num] / (FP[num] + TN[num])
    #     print("threshold : {:.2f}, TP : {:.2f}, FP : {:.2f}, TN : {:.2f}, FP : {:.2f}, TPR : {:.2f}, FPR : {:.2f}"
    #           .format(b, TP[num], FP[num], TN[num], FN[num], TPR[num], FPR[num]))

    print("AUROC score : ")
    print(roc_auc_score(np.hstack((normal_label, abnormal_label)), np.hstack((normal_abs, abnormal_abs))))

    # normal_abs = (normal_abs-min(normal_abs))/(max(normal_abs) - min(normal_abs))
    # abnormal_abs = (abnormal_abs-min(abnormal_abs))/(max(abnormal_abs) - min(abnormal_abs))
    fpr, tpr, thres = roc_curve(np.hstack((normal_label, abnormal_label)), np.hstack((normal_abs, abnormal_abs)), pos_label=1)
    print(auc(fpr, tpr))

    plt.figure(figsize=(10, 6))
    plt.subplot(121)
    plt.plot(fpr, tpr)
    plt.xlabel("fpr")
    plt.ylabel("tpr")

    # plt.subplot(122)
    # plt.plot(FPR, TPR)
    # plt.xlabel("FPR")
    # plt.ylabel("TPR")

    # plt.show()

    plt.figure(figsize=(10, 12))
    # plt.subplot(121)
    # plt.plot(a, TP)
    # plt.plot(a, FN)
    # plt.plot(a, TN)
    # plt.plot(a, FP)
    # plt.legend(["TP", "FN", "TN", "FP"])
    bins = np.linspace(600,2500, 100)
    #
    plt.subplot(122)
    plt.hist([normal_abs, abnormal_abs], bins, label=['normal', 'abnormal'])
    plt.legend(loc="upper right")
    plt.title("Normalized Distribution")
    plt.show()




    #
    # checkpoint = torch.load(os.path.join(save_model, fol, "checkpoint_last.pth.tar"))
    # if hidden_size != checkpoint["hidden_size"]:
    #     raise AttributeError("checkpoint's hidden size is not same")
    # ##### load model
    # model.load_state_dict(checkpoint["model"])
    #
    # a = load_data.MVTec_dataloader(image_dir=os.path.join("/media/seonghun/data1/MVTec", "cable"), mode="test", in_size=256)
    # train_loader = torch.utils.data.DataLoader(a, batch_size=1, shuffle=True, num_workers=2)
    #
    #
    # # with torch.no_grad():
    # for i, (org_image, _) in enumerate(train_loader):
    #     org_image = org_image.to(device)
    #     model = model.to(device)
    #
    #     out, _, _ = model(org_image)
    #     out = torch.tanh(out).squeeze().cpu().detach().numpy()
    #     print(out.transpose((1,2,0)))
    #     cv2.imshow("123",out.transpose((1,2,0)))
    #
    #     # trans = transforms.ToPILImage()
    #     # plt.imshow(trans(out.squeeze().cpu().detach()))
    #     # plt.show()
    #     cv2.waitKey()
    #
    #     break


    # n=5
    #
    #
    # model = models.VAE(hidden_size, input_size)
    # model = model.to(device)
    # checkpoint = torch.load(os.path.join(save_model, fol, "checkpoint_last.pth.tar"))
    # model.load_state_dict(checkpoint["model"])
    #
    #
    # x_axis = np.linspace(-3, 3, 5)
    # y_axis = np.linspace(-3, 3, 5)



    # vec2recon(model, hidden_size)
