import datetime
import glob
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

img_folder = "/mnt/Create_multi_mask/img/"
mask_folder = "/mnt/Create_multi_mask/mask_one/"
results_folder = "/home/owner/PycharmProjects/single_unet/results/"
structure_number = 1


# skip_str = [6, 30, 31]
# skip_str = [31,32, 33, 34]

def get_filelist():
    img_list = sorted(glob.glob(img_folder + "*"))
    mask_list = sorted(glob.glob(mask_folder + "*"))
    return img_list, mask_list


class UNet(nn.Module):

    def __init__(self, in_channels=1, out_channels=1, init_features=16):
        super(UNet, self).__init__()

        features = init_features
        self.encoder1 = UNet._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2, output_padding=1
        )
        self.decoder4 = UNet._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        # return torch.sigmoid(self.conv(dec1))
        return self.conv(dec1)

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )


class DiceLoss(nn.Module):

    def __init__(self):
        super(DiceLoss, self).__init__()
        self.smooth = 1.0

    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size()
        y_pred = y_pred[:, 0].contiguous().view(-1)
        y_true = y_true[:, 0].contiguous().view(-1)
        torch_one = torch.ones(len(y_pred)).to("cuda")
        torch_zero = torch.zeros(len(y_pred)).to("cuda")
        y_pred = torch.where(y_pred > 0.5, torch_one, torch_zero)
        intersection = (y_pred * y_true).sum()
        dsc = (2. * intersection + self.smooth) / (y_pred.sum() + y_true.sum() + self.smooth)
        return 1. - dsc


class MultiDiceLoss(nn.Module):

    def __init__(self):
        super(MultiDiceLoss, self).__init__()
        self.smooth = 1.0

    def forward(self, y_pred, y_true):
        for some_channel in range(y_pred.shape[2]):
            print('a')
        return 0


def iou_score(output, target, in_loss_func=False):
    smooth = 1e-5
    if not in_loss_func:
        if torch.is_tensor(output):
            output = torch.sigmoid(output).cpu().detach().numpy().copy()
        if torch.is_tensor(target):
            target = target.data.cpu().detach().numpy().copy()
    output = output > 0.5
    target = target > 0.5
    intersection = (output & target).sum()
    union = (output | target).sum()

    return (intersection + smooth) / (union + smooth)


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super(BCEDiceLoss, self).__init__()

    def forward(self, input, target):
        the_loss_func = 0.0
        for the_channels in range(input.shape[1]):
            label_in_a_channel = torch.where(target == the_channels, 1.0, 0.0)
            pred_in_a_channel = input[:, the_channels, :, :]
            bce = F.binary_cross_entropy_with_logits(pred_in_a_channel, label_in_a_channel)
            smooth = 1e-5
            pred_in_a_channel = torch.sigmoid(pred_in_a_channel)
            num = label_in_a_channel.size(0)
            pred_in_a_channel = pred_in_a_channel.view(num, -1)
            label_in_a_channel = label_in_a_channel.view(num, -1)
            intersection = (pred_in_a_channel * label_in_a_channel)
            dice = (2. * intersection.sum(1) + smooth) / (pred_in_a_channel.sum(1) + label_in_a_channel.sum(1) + smooth)
            dice = 1 - dice.sum() / num
            the_loss_func += 0.5 * bce + dice
        return the_loss_func


class DiceonlyLoss(nn.Module):
    def __init__(self):
        super(DiceonlyLoss, self).__init__()

    def forward(self, input, target):
        the_loss_func = []
        for the_channels in range(input.shape[1]):
            label_in_a_channel = torch.where(target == the_channels, 1.0, 0.0)
            pred_in_a_channel = input[:, the_channels, :, :]
            smooth = 1e-5
            pred_in_a_channel = torch.sigmoid(pred_in_a_channel)
            num = label_in_a_channel.size(0)
            pred_in_a_channel = pred_in_a_channel.view(num, -1)
            label_in_a_channel = label_in_a_channel.view(num, -1)
            intersection = (pred_in_a_channel * label_in_a_channel)
            dice = (2. * intersection.sum(1) + smooth) / (pred_in_a_channel.sum(1) + label_in_a_channel.sum(1) + smooth)
            dice = dice.to('cpu').detach().numpy().copy()
            the_loss_func.append(np.sum(dice) / num)

        return the_loss_func


class multi_class_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, preds, trues):
        return torch.sum(torch.pow((preds - trues), 2))


class my_dataset(Dataset):
    def __init__(self, img_list, mask_list, transform=None):
        self.img_list = img_list
        self.mask_list = mask_list

    def __getitem__(self, index):
        path = self.img_list[index]
        img = np.load(path)
        mask = np.load(mask_folder + path.split("/")[-1])
        return img, mask

    def __len__(self):
        return len(self.img_list)


def main():
    batchsize = 6
    num_epochs = 10
    learning_rate = 0.0001
    loss_list, iteration_list, accuracy_list = [], [], []
    count = 0
    imgs, masks = get_filelist()
    train_x, test_x, train_y, test_y = train_test_split(imgs, masks, test_size=0.01, random_state=22)
    train_dataset = my_dataset(train_x, train_y)
    test_dataset = my_dataset(test_x, test_y)
    dataloader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batchsize, shuffle=True, num_workers=4, pin_memory=True)
    # data_iter = iter(dataloader)
    # timgs, tlabels = data_iter.next()

    unet = UNet(in_channels=1, out_channels=34).to("cuda")
    bce_loss = BCEDiceLoss()
    ver_dice_loss = DiceonlyLoss()
    # dsc_loss = multi_class_loss()
    # celoss = nn.CrossEntropyLoss()
    # dsc_loss = DiceLoss()
    # mdcloss = MultiDiceLoss()

    optimizer = torch.optim.Adam(unet.parameters(), lr=learning_rate, weight_decay=0.001)
    # optimizer = torch.optim.SGD(unet.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0001)
    torch.backends.cudnn.benchmark = True
    for epoch in range(num_epochs):
        torch.save(unet.state_dict(), "unet.pth")
        for img, mask in dataloader:
            # plt.figure()
            # plt.imshow(img[0, :, :], cmap="Greys")
            # plt.imshow(mask[0, 30, :, :], alpha=0.5)
            # plt.show()
            unet.train()
            img = img.view(-1, 1, 600, 600)
            mask = mask.view(-1, 1, 600, 600)
            img_tnsr = img.float().requires_grad_(True).cuda()
            mask_tnsr = mask.float().requires_grad_(True).to("cuda")
            optimizer.zero_grad()
            y_pred = unet(img_tnsr)
            # loss = dsc_loss(y_pred, mask_tnsr)
            loss = bce_loss(y_pred, mask_tnsr.view(-1, 600, 600).long())
            loss.backward()
            optimizer.step()
            count += 1
            if count % 100 == 0:
                unet = unet.eval()
                for j, (timg, tmask) in enumerate(test_loader):
                    timg = timg.clone().detach().reshape(-1, 1, 600, 600)
                    tmask = tmask.clone().detach().reshape(-1, 1, 600, 600)
                    timg_tnsr = timg.float().cuda()
                    tmask_tnsr = tmask.float().to("cuda")
                    t_pred = unet(timg_tnsr)
                    # tloss = iou_score(t_pred, tmask_tnsr)
                    # tloss = dsc_loss(t_pred, tmask_tnsr)
                    tloss = ver_dice_loss(t_pred, tmask_tnsr.view(-1, 600, 600).long())
                    tloss = np.array(tloss)
                    if j == 0:
                        tlosss = tloss
                    else:
                        tlosss += tloss

                    timg = timg.view(-1, 1, 600, 600).to("cpu").detach().numpy()
                    tmask = tmask.view(-1, 1, 600, 600).to("cpu").detach().numpy()
                    # t_pred = t_pred.view(-1, 1,  600, 600).to("cpu").detach().numpy()
                    # val_dice_loss = mdcloss(t_pred, tmask)
                    # val_iou_loss = iou_score(t_pred, tmask)
                    fig = plt.figure(figsize=(20, 12))
                    ax1 = fig.add_subplot(121)
                    ax2 = fig.add_subplot(122)
                    ax1.set_axis_off()
                    ax2.set_axis_off()
                    # print(timg.shape)
                    # t_pred = np.where(t_pred > 0.5, 1, 0)
                    ax1.imshow(timg[0, 0, :, :], cmap="gray")
                    _, idx = torch.max(t_pred, dim=1)
                    idx = idx.to("cpu").detach().numpy()
                    # print(idx.shape())
                    ax1.imshow(idx[0, :, :], alpha=0.5, cmap="jet", vmin=0.5, vmax=35)
                    ax2.imshow(timg[0, 0, :, :], cmap="gray")
                    ax2.imshow(tmask[0, 0, :, :], alpha=0.5, cmap="jet", vmin=0.5, vmax=35)
                    fig.savefig(results_folder + str(j) + ".png")
                    plt.close()
                tlosss = tlosss / (j + 1)

                print("Epoch {}, Iter. {}, Loss {:.8f}".format(epoch, count, loss.data))
                # print(tlosss)
                np.savetxt(results_folder + "dices.txt", tlosss)


if __name__ == "__main__":
    print(datetime.datetime.now())
    main()
    print(datetime.datetime.now())
