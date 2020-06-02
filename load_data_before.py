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

# if __name__ == '__main__':
#     a = cifar100_dataloader(imagedir="/media/seonghun/data1/CIFAR100", mode="fine/train", anomally=None)
#     trainloader = DataLoader(a,
#         batch_size=512, shuffle=True, num_workers=2)
#
#     print(b)

