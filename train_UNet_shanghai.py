# --------------------------------------------------------
# Copyright (c) 2018 Fudan-VTS
# Licensed under The MIT License [see LICENSE for details]
# Written by liwenxi
# --------------------------------------------------------
import copy
import math
import time

import torch
from tensorboardX import SummaryWriter
from torch import nn, optim
from torchvision import transforms

from datasets.ShanghaiTechDataset import ShanghaiTechDataset
from networks.TutorNet import TutorNet
from networks.UNet import U_Net
from utils import mse_loss, auto_loss


def main():
    writer = SummaryWriter()
    num_epochs = 1000
    batch_size = 1

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = ShanghaiTechDataset(mode="train", transform=transform)
    dataset_test = ShanghaiTechDataset(mode="test", transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=20)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=20)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = U_Net().to(device)
    lossnet = TutorNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    optimizer2 = optim.Adam(lossnet.parameters(), lr=1e-4)

    MSEloss = nn.MSELoss(reduction='sum').to(device)
    L1loss = nn.L1Loss().to(device)

    sumMAE_best = 90

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        start_time = time.time()
        for phase in ['train', 'test']:
            print("strating Itrerate")
            running_loss = torch.tensor(0.0).cuda()
            running_loss_s = torch.tensor(0.0).cuda()
            train_start_time = time.time()
            sum_MAE = 0
            if phase == 'train':
                model.train()  # Set model to training mode

                step = 0
                for rgb, ground_truth, _ in dataloader:
                    step_time = time.time()
                    print("Epoch {} Train Step {}: ".format(epoch, step))
                    step += 1
                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        if phase == 'train':
                            rgb = rgb.float().to(device)
                            ground_truth = ground_truth.float().to(device)

                            outputs1 = model(rgb)
                            x_w = lossnet(rgb)
                            loss = mse_loss(outputs1.squeeze(), ground_truth.squeeze())
                            loss_w = torch.sum(loss)
                            loss_w.backward(retain_graph=True)
                            optimizer.step()

                            optimizer2.zero_grad()

                            loss_s = auto_loss(x_w, loss, lossnet.parameters(),
                                               M=0.8, alpha=0)
                            print("Loss: ", loss_w.item())
                            loss_s.backward()

                            optimizer2.step()

                    running_loss += loss_w.item()
                    running_loss_s += loss_s.item()
                    print("This Step Used", time.time() - step_time)
                writer.add_scalar('scalar/Loss_w', running_loss / dataset.__len__(), epoch)
                writer.add_scalar('scalar/Loss_s', running_loss_s / dataset.__len__(), epoch)
                print("This Train Used", time.time() - train_start_time)

            else:
                print("starting test")
                model.eval()
                torch.set_grad_enabled(False)
                step = 0
                MAE = 0
                MSE = 0

                for rgb, ground_truth, _ in dataloader_test:
                    step_time = time.time()
                    print("Epoch {} Test Step {}: ".format(epoch, step))
                    step += 1

                    rgb = rgb.float().to(device)
                    ground_truth = ground_truth.float().to(device)

                    outputs1 = model(rgb)

                    outputs1 = torch.sum(outputs1 / 1000, (-1, -2))

                    ground_truth = torch.sum(ground_truth / 1000, (-1, -2))
                    MAE += L1loss(outputs1.squeeze(), ground_truth.squeeze())
                    MSE += MSEloss(outputs1.squeeze(), ground_truth.squeeze())

                    print("MAE", MAE.item() / step)
                    print("MSE", math.sqrt(MSE.item() / step))
                    print("This Step Used", time.time() - step_time)
                sum_MAE += MAE.item() / dataset_test.__len__() * batch_size
                print("MAE", MAE.item() / dataset_test.__len__() * batch_size)
                print("MSE", math.sqrt(MSE.item() / dataset_test.__len__() * batch_size))
                writer.add_scalar('scalar/MAE', MAE.item() / dataset_test.__len__() * batch_size, epoch)
                writer.add_scalar('scalar/MSE', math.sqrt(MSE.item() / dataset_test.__len__()), epoch)
                print("This Test Used", time.time() - train_start_time)

                if sum_MAE < sumMAE_best:
                    best_model_wts = copy.deepcopy(model.state_dict())
                    sumMAE_best = sum_MAE
                    torch.save(best_model_wts, "unet_tutor_unet_shanghai_" + "MAE" + str(sum_MAE)
                               + 'MSE' + str(
                        math.sqrt(MSE.item() / dataset_test.__len__() * batch_size)) + 'best_model_wts.pkl')

                    best_model_wts = copy.deepcopy(lossnet.state_dict())
                    torch.save(best_model_wts, "unet_lossnet_shanghai_" + "MAE" + str(sum_MAE)
                               + 'MSE' + str(
                        math.sqrt(MSE.item() / dataset_test.__len__() * batch_size)) + 'best_model_wts.pkl')

        print("This Epoch used", time.time() - start_time)
        print()


if __name__ == '__main__':
    main()
