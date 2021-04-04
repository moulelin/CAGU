# -*- coding: utf-8 -*-
'''
@File  : train.py
@Author: Moule Lin
@Date  : 2021/1/6 8:41
@Github: https://github.com/moulelin
'''

import torch
import torch.nn as nn
from model.CAGU import CAGU
from model.load_image import getImage
import torch.optim as optim
from torch.utils.data import DataLoader
import warnings
import numpy as np
import argparse
warnings.filterwarnings('ignore')
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
import os
import math
from metrics import AverageMeter,Evaluator
parser = argparse.ArgumentParser(description="")
parser.add_argument("--epochs",type=int,default=1001)
parser.add_argument("--lr",type=float,default=0.0005)
parser.add_argument("--gpu_ids",type=int,default=0)
parser.add_argument("--random-seed",type=int,default=1)
parser.add_argument("--image_path",type=str,default="Pavia.mat")
parser.add_argument("--label_path",type=str,default="Pavia_gt.mat")
parser.add_argument("--check_point",type=str,default="model")
args = parser.parse_args()
torch.manual_seed(args.random_seed)
torch.cuda.manual_seed(args.random_seed)
np.random.seed(args.random_seed)
torch.cuda.manual_seed_all(args.random_seed)
torch.backends.cudnn.enable = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

trans_image = transforms.Compose([
    transforms.ToTensor(),
])
trans_label = transforms.Compose([
    transforms.ToTensor()
])
train = getImage("dataset/Houston_2018/")
trainDataset = DataLoader(train, batch_size=5, shuffle=False, num_workers=2, drop_last=False,pin_memory=False)


def train():
    trainloss = {}
    oa= {}
    aa={}
    kappa_= {}
    torch.cuda.current_device()
    device = torch.device("cuda:{}".format(args.gpu_ids) if torch.cuda.is_available() else "cpu")
    net = CAGU(50,20)

    net = net.to(device)

    net.train()
    criterion = nn.CrossEntropyLoss()
    lambdal = lambda epoch:pow((1-epoch/args.epochs),0.9)
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-8)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambdal)
    best_monitor = -math.inf
    for epoch in range(args.epochs):
        avg_loss = AverageMeter() # average loss
        avg_miou = AverageMeter() # average iou
        pixel_accuracy = AverageMeter() # pixel_accuracy
        precision = AverageMeter()  # precision
        recall = AverageMeter() # Recall
        kappa = AverageMeter()

        for i,data in enumerate(trainDataset,0):
            inputs, labels = data
            inputs,labels = inputs.to(device),labels.to(device)
            # labels = labels.to(torch.long)
            outputs = net(inputs,labels)
            evaluator = Evaluator(20)
            loss = criterion(outputs, labels) # outputs->1*9*W*H labels->1*1#W*H
            evaluator.add_batch(labels,outputs) # confusion matrix, saving in instance of evaluator
            miou = evaluator.Mean_Intersection_over_Union() # get mean iou by confusion matrix
            avg_loss.add(loss.data.item(),inputs.size(0)) # get mean loss saving in avg_loss
            avg_miou.add(miou,inputs.size(0)) # get mean iou saving in avg_loss

            accuracy = evaluator.Pixel_Accuracy()


            pixel_accuracy.add(accuracy,inputs.size(0))

            precision_,_ = evaluator.Precision()
            precision.add(precision_,inputs.size(0))

            recall_,_ = evaluator.Recall()
            recall.add(recall_,inputs.size(0))

            kappa_value = evaluator.Kappa()
            kappa.add(kappa_value,inputs.size(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(loss.item())
            # if i%5==0 and i:
            #     print(f"temp loss is {loss.item()}")
        scheduler.step(epoch)
      #  if (epoch%10 == 0 and epoch) or epoch+1 == args.epochs:


        print("Epoch: {}, Loss: {}, mIoU：{},AA:{},OA:{},Kappa:{}".format(epoch,avg_loss.average, avg_miou.average,pixel_accuracy.average,recall.average,kappa.average))
        trainloss[epoch]=(avg_loss.average)
        oa[epoch]=(pixel_accuracy.average)
        aa[epoch]=(recall.average)
        kappa_[epoch]=(kappa.average)


        if best_monitor<avg_miou.average:
            checkpoint={
                'model_dict': net.state_dict(),
               # 'iter_num':epoch,
               # 'optimizer':optimizer.state_dict()
            }
            torch.save(checkpoint,f"model_houston_2018/checkpoing")
            best_monitor = avg_miou.average
        if epoch and epoch%10 == 0:
            checkpoint = {
                'model_dict': net.state_dict(),
                # 'iter_num':epoch,
                # 'optimizer':optimizer.state_dict()
            }
            torch.save(checkpoint, f"model_houston_2018/checkpoing{epoch}")

        loss_value = open("2018/loss.txt",'w')
        loss_value.write(str(trainloss))
        loss_value.close()

        oa_value = open("2018/oa.txt", 'w')
        oa_value.write(str(oa))
        oa_value.close()


        aa_value = open("2018/aa.txt", 'w')
        aa_value.write(str(aa))
        aa_value.close()

        kappa_value = open("2018/kappa.txt", 'w')
        kappa_value.write(str(kappa))
        kappa_value.close()

if __name__=='__main__':
    train()
