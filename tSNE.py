import torch
import os
import numpy as np
import math

import sklearn
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import pandas as pd
import seaborn as sb

import models_VAE as model
import load_data


def plot(x, colors, num_labels):

    palette = np.array(sb.color_palette("hls", num_labels))  #Choosing color palette

    # Create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40, c=palette[colors.astype(np.int)])
    # Add the labels for each digit.
    txts = []
    for i in range(num_labels):
        # Position of each label.
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([pe.Stroke(linewidth=5, foreground="w"), pe.Normal()])
        txts.append(txt)
    return f, ax, txts


if __name__ == '__main__':

    img_dir = "/media/seonghun/data1/MVTec"
    sample_dir = "sample"
    save_model = "./VAE_MVTec_model"
    save_result = "./result"
    img_fol_name = "wood"
    mode = "train_one"
    input_size = 256*256
    input_size = int(math.sqrt(input_size))
    batch_size = 8
    hidden_size = [64, 128, 256, 512, 800]
    save_model = "./VAE_MVTec_model"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    model = model.VAE(hidden_size, input_size)
    checkpoint = torch.load(os.path.join(save_model, "wood_VAE.pth.tar"))
    model.load_state_dict(checkpoint["model"])


    fol = ""
    for i in hidden_size:
        fol+=str(i)


    normal_data = load_data.MVTec_dataloader(image_dir=os.path.join(img_dir,img_fol_name), mode=mode, in_size=input_size)
    train_loader = torch.utils.data.DataLoader(normal_data, batch_size=batch_size, shuffle=True, num_workers=2)
    abnormal_data = load_data.MVTec_dataloader(image_dir=os.path.join(img_dir,img_fol_name), mode="test", in_size=input_size)
    anomaly_loader = torch.utils.data.DataLoader(abnormal_data, batch_size=batch_size, shuffle=True, num_workers=2)
    test_normal_loader = torch.utils.data.DataLoader(load_data.MVTec_dataloader(image_dir = os.path.join(img_dir, img_fol_name), mode="test_normal", in_size = input_size)
                                                   , batch_size=batch_size, shuffle=True, num_workers=2)


    with torch.no_grad():
        for idx, (org_image, gt) in enumerate(train_loader):
            org_image = org_image.to(device)
            model.to(device).eval()
            mu, logvar = model.encoder(org_image)
            if idx == 0:
                X = model.reparameterize(mu, logvar)
            else:
                X = torch.cat((X, model.reparameterize(mu, logvar)), dim=0)
        tmp_Y = torch.zeros(X.shape[0])

        for idx, (org_image, gt) in enumerate(test_normal_loader):
            org_image = org_image.to(device)
            model.to(device).eval()
            mu, logvar = model.encoder(org_image)
            if idx == 0:
                tmp_X = model.reparameterize(mu, logvar)
            else:
                tmp_X = torch.cat((tmp_X, model.reparameterize(mu, logvar)), dim=0)
        # X = tmp_X
        X = torch.cat((X, tmp_X), dim=0)
        # Y = torch.zeros(tmp_X.shape[0])
        Y = torch.cat((tmp_Y, torch.ones(tmp_X.shape[0])))

        for idx, (org_image, gt) in enumerate(anomaly_loader):
            org_image = org_image.to(device)
            model.to(device).eval()
            mu, logvar = model.encoder(org_image)
            if idx == 0:
                tmp_X = model.reparameterize(mu, logvar)
            else:
                tmp_X = torch.cat((tmp_X, model.reparameterize(mu, logvar)), dim=0)
        X = torch.cat((X, tmp_X), dim=0).squeeze().cpu().detach().numpy()
        Y = torch.cat((Y,2*torch.ones(tmp_X.shape[0]))).cpu().detach().numpy()



        X = StandardScaler().fit_transform(X)
        pca = PCA(n_components=2)
        principalComponents = pca.fit_transform(X)
        principalDf = pd.DataFrame(data=principalComponents
                                   , columns=['principal component 1', 'principal component 2'])

        Y = np.expand_dims(Y, axis = 1)
        df = pd.DataFrame(Y)
        finalDf = pd.concat([principalDf, df], axis=1)
        finalDf.columns = ["principalcomponent1", "principalcomponent2", "label"]
        flatui = ["red", "green", "blue"]
        sb.set_palette(flatui)
        sb.lmplot(x="principalcomponent1", y="principalcomponent2", data=finalDf, fit_reg=False,hue="label", legend=False)

        plt.show()


        # TSNE_final = TSNE(perplexity=20).fit_transform(X)
        #
        # plot(TSNE_final, Y, 3)
        # plt.show()