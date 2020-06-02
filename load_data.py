import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]


class MNIST(Dataset):
    def __init__(self, imagedir, mode):
        self.imgdir = os.path.join(imagedir, mode)
        self.gtlist = sorted(os.listdir(self.imgdir))
        self.imglist = []
        for fol in self.gtlist:
            self.imglist.extend(sorted(listdir_fullpath(os.path.join(self.imgdir, fol))))
        self.gt = []
        for file in self.imglist:
            self.gt.append(file[file[:file.rfind("/")].rfind("/")+1:file.rfind("/")])
        self.len = len(self.imglist)
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
        ])
    def __len__(self):
        return len(self.imglist)

    def get_gtlist(self):
        return self.gtlist

    def __getitem__(self, idx):
        img = Image.open(self.imglist[idx]).convert("L")
        img = self.preprocess(img)
        return img, self.gt[idx]



class cifar10_dataloader(Dataset):
    def __init__(self, imagedir, mode):
        self.imgdir = os.path.join(imagedir, mode)
        self.imglist = sorted(os.listdir(self.imgdir))
        self.gtlist = []
        if 'train' in mode:
            self.preprocess = transforms.Compose([
                transforms.ToTensor(),
            ])
        else:
            self.preprocess = transforms.Compose([
                transforms.ToTensor(),
            ])

        f = open(os.path.join(imagedir, "labels.txt"), "r")
        while True:
            line = f.readline()
            if not line: break
            self.gtlist.append(line[:-1])
        f.close()

    def get_gtlist(self):
        return self.gtlist

    def __len__(self):
        return len(self.imglist)

    def __getitem__(self, idx):
        gt = []
        for gt_name in self.gtlist:
            if gt_name in self.imglist[idx]:
                gt = self.gtlist.index(gt_name)
                break
        img = Image.open(os.path.join(self.imgdir, self.imglist[idx])).convert("RGB")
        img = self.preprocess(img)
        return img, gt

class MVTec_dataloader(Dataset):
    def __init__(self, image_dir, mode, in_size):
        self.gtlist = sorted(os.listdir(image_dir))
        self.mode = mode
        self.imglist = []
        self.gt = []

        if mode == "train":
            mode += "/good"
            for gt in self.gtlist:
                self.gt.extend(self.gtlist * len(os.path.join(image_dir, gt, mode)))
                self.imglist.extend(sorted(listdir_fullpath(os.path.join(image_dir, gt, mode))))
        elif self.mode == "train_one":
            self.gt = image_dir[image_dir.rfind("/"):]
            self.imglist.extend(sorted(listdir_fullpath(os.path.join(image_dir, "train/good"))))
        elif "test" in self.mode:
            self.test_list = os.listdir(os.path.join(image_dir, self.mode))
            print(self.test_list)
            for list in self.test_list:
                self.imglist.extend(sorted(listdir_fullpath(os.path.join(image_dir, self.mode, list))))
                self.gt.extend(list*len(sorted(listdir_fullpath(os.path.join(image_dir, self.mode, list)))))

        if "train" in mode:
            self.preprocess = transforms.Compose([
                transforms.Resize([in_size,in_size]),
                # transforms.Pad(8,padding_mode="symmetric"),
                # transforms.RandomCrop((in_size,in_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5,0.5,0.5], std = [0.5, 0.5, 0.5]),
            ])
        else:
            self.preprocess = transforms.Compose([
                transforms.Resize([in_size,in_size]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5,0.5,0.5], std = [0.5, 0.5, 0.5]),
            ])

    def __len__(self):
        return len(self.imglist)

    def __getitem__(self, idx):
        img = self.preprocess(Image.open(self.imglist[idx]).convert("RGB"))
        gt = self.gt if self.mode == "train_one" else self.gt[idx]
        return img, gt


class cifar_anomaly(Dataset):
    def __init__(self, imagedir, mode):
        self.imgdir = imagedir
        self.imglist = sorted(os.listdir(self.imgdir))
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
        ])
        if 'normal' == mode:
            self.gt = "normal"
        else:
            self.gt = "abnormal"

    def __len__(self):
        return len(self.imglist)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.imgdir, self.imglist[idx])).convert("RGB")
        img = self.preprocess(img)
        return img, self.gt



# if __name__ == '__main__':
#     a = cifar100_dataloader(imagedir="/media/seonghun/data1/CIFAR100", mode="fine/train", anomally=None)
#     trainloader = DataLoader(a,
#         batch_size=512, shuffle=True, num_workers=2)
#
#     print(b)

