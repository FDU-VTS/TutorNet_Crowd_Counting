# --------------------------------------------------------
# SANet
# Copyright (c) 2018 Fudan-VTS
# Licensed under The MIT License [see LICENSE for details]
# Written by liwenxi
# --------------------------------------------------------
import copy
# from SANet import SANet
# from blaze_model import BlazeFace
import math
import time

import pytorch_ssim
import torch
from Fudan_dataset import FudanDataset
# from piexl_net import *
from Res101 import Res101, Res101_main
# from WorldExpoDataset import WorldExpoDataset, WorldExpoTestDataset
from baseline_de import mse_loss, auto_loss
from tensorboardX import SummaryWriter
from torch import nn, optim
from torchvision import transforms
from unet import *


def main():
    writer = SummaryWriter()
    # writer = SummaryWriter('tensorboard.log')
    num_epochs = 1000
    batch_size = 1
    # img_path = "./mall_dataset/frames/

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = FudanDataset(mode="train", transform=transform)
    dataset_test = FudanDataset(mode="test", transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=20)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=20)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device2 = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # model = Baseline_densenet(16).to(device)
    model = Res101_main().to(device)
    # model = CSRNet().to(device)
    # model = U_Net().to(device)
    # model = VGG().to(device)
    # lossnet = LossNet(16).to(device)
    lossnet = Res101().to(device2)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    # optimizer = optim.SGD(model.parameters(), lr=1e-5, momentum=0.95, weight_decay=5 * 1e-5)
    optimizer2 = optim.Adam(lossnet.parameters(), lr=1e-4)

    # criterion = mse_loss()

    MSEloss = nn.MSELoss(reduction='sum').to(device)
    L1loss = nn.L1Loss().to(device)
    BCEloss = nn.BCELoss().to(device)
    SmoothL1 = nn.SmoothL1Loss(reduction='sum').to(device)
    SSIM_loss = pytorch_ssim.SSIM()

    sumMAE_best = 90
    # best_model_wts = copy.deepcopy(model.state_dict())
    lossoutput = 0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        start_time = time.time()
        # for phase in ['test', 'train']:

        for phase in ['train', 'test']:
            print("strating Itrerate")
            running_loss = torch.tensor(0.0).cuda()
            running_loss_s = torch.tensor(0.0).cuda()
            train_start_time = time.time()
            MAE = 0
            MSE = 0
            sum_MAE = 0
            if phase == 'train':
                model.train()  # Set model to training mode

                step = 0;
                for rgb, ground_truth, _ in dataloader:
                    step_time = time.time()
                    print("Epoch {} Train Step {}: ".format(epoch, step))
                    step += 1

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        if phase == 'train':
                            rgb = rgb.float().to(device)
                            # flow = rgb.clone()
                            ground_truth = ground_truth.float().to(device)

                            # flow.fill_(1)
                            outputs1 = model(rgb)
                            x_w = lossnet(rgb.to(device2))
                            # print(outputs2.size(), mask.size())
                            loss = mse_loss(outputs1.squeeze(), ground_truth.squeeze())
                            loss_w = torch.sum(loss * x_w.to(device))
                            loss_w.backward(retain_graph=True)
                            optimizer.step()
                            # print(MSEloss(outputs1.squeeze(), ground_truth.squeeze()))
                            # print(loss_w)
                            optimizer2.zero_grad()
                            # loss_s = snet_loss(x_w, torch.clamp(1-1/(loss+1e-5), min=0), lossnet.parameters(), M=1, alpha=0)
                            # loss_s = snet_loss(x_w, loss, lossnet.parameters(),
                            #                    M=2, alpha=0)
                            # loss_s = snet_loss(x_w, loss, lossnet.parameters(),
                            #                    M=17, alpha=0)
                            # loss_s = auto_loss(x_w, torch.clamp(1/(loss+1e-5), min=0, max=1),lossnet.parameters(),
                            #                    M=1, alpha=0)
                            loss_s = auto_loss(x_w, loss.to(device2), lossnet.parameters(),
                                               M=1, alpha=0)
                            # loss = criterion(outputs3.squeeze(), ground_truth.squeeze())

                            print("Loss: ", loss_s.item())
                            print("Loss: ", loss_w.item())

                            loss_s.backward()

                            optimizer2.step()

                    running_loss += loss_w.item()
                    running_loss_s += loss_s.item()
                    # print("Loss: ", running_loss/step)
                    print("This Step Used", time.time() - step_time)
                writer.add_scalar('scalar/Loss_w', running_loss / dataset.__len__(), epoch)
                writer.add_scalar('scalar/Loss_s', running_loss_s / dataset.__len__(), epoch)
                print("This Train Used", time.time() - train_start_time)

            else:
                print("starting test")
                # rgb_net.eval()
                # flow_net.eval()
                # gate_net.eval()
                model.eval()
                torch.set_grad_enabled(False)
                step = 0;
                # train_start_time = time.time()
                MAE = 0
                MSE = 0

                for rgb, ground_truth, _ in dataloader_test:
                    step_time = time.time()
                    print("Epoch {} Test Step {}: ".format(epoch, step))
                    step += 1

                    rgb = rgb.float().to(device)
                    # flow = rgb.clone()
                    ground_truth = ground_truth.float().to(device)

                    # flow.fill_(1)
                    outputs1 = model(rgb)

                    outputs1 = torch.sum(outputs1 / 1000, (-1, -2))

                    # outputs3 = torch.sum(outputs3 / 100, (-1, -2))

                    ground_truth = torch.sum(ground_truth / 1000, (-1, -2))
                    MAE += L1loss(outputs1.squeeze(), ground_truth.squeeze())
                    MSE += MSEloss(outputs1.squeeze(), ground_truth.squeeze())

                    print("MAE", MAE.item() / step)
                    print("MSE", math.sqrt(MSE.item() / step))
                    # writer.add_scalar('scalar/MAE', MAE.item()/step, step)
                    print("This Step Used", time.time() - step_time)
                sum_MAE += MAE.item() / dataset_test.__len__() * batch_size
                print("MAE", MAE.item() / dataset_test.__len__() * batch_size)
                print("MSE", math.sqrt(MSE.item() / dataset_test.__len__() * batch_size))
                writer.add_scalar('scalar/MAE', MAE.item() / dataset_test.__len__() * batch_size, epoch)
                writer.add_scalar('scalar/MSE', math.sqrt(MSE.item() / dataset_test.__len__() * batch_size), epoch)

                # writer.add_scalar('scalar/MAE1', MAE1.item() / dataset_test.__len__() * batch_size, epoch)
                # writer.add_scalar('scalar/MAE2', MAE2.item() / dataset_test.__len__() * batch_size, epoch)
                # # writer.add_scalar('scalar/MAE3', MAE3.item() / dataset_test.__len__() * batch_size, epoch)
                # # writer.add_scalar('scalar/scene1MSE', math.sqrt(MSE.item()/dataset_test1.__len__()*batch_size), epoch)
                print("This Test Used", time.time() - train_start_time)

                if sum_MAE < sumMAE_best:
                    best_model_wts = copy.deepcopy(model.state_dict())
                    sumMAE_best = sum_MAE
                    torch.save(best_model_wts, "CSR_fudan_" + "MAE" + str(sum_MAE)
                               + 'MSE' + str(
                        math.sqrt(MSE.item() / dataset_test.__len__() * batch_size)) + 'best_model_wts.pkl')

                    best_model_wts = copy.deepcopy(lossnet.state_dict())
                    sumMAE_best = sum_MAE
                    torch.save(best_model_wts, "lossnet_fudan_" + "MAE" + str(sum_MAE)
                               + 'MSE' + str(
                        math.sqrt(MSE.item() / dataset_test.__len__() * batch_size)) + 'best_model_wts.pkl')

        print("This Epoch used", time.time() - start_time)
        print()


if __name__ == '__main__':
    main()
