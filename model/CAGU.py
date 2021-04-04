#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File  : CAGU.py
@Author: Moule Lin
@Date  : 2020/8/26 12:08

'''
from .parts_unet import *
from .layers_chebnet import ChebConv
from .layers_gcn import GCN
import numpy as np
import torch
class GRU(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers):
        super(GRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.outer = nn.Linear(hidden_size, input_size) # 保证输出的维度信息不变
        self.drop = nn.Dropout(p=0.5)
    def forward(self,x):
        # if hidden_paramenter_get_in is None:
        #     hidden_paramenter_get_in = torch.randn(x.size(0),x.size(1),2*x.size(2)).requires_grad_()
        # # x -->[num_node，Batch, feature]
        out, hidden_parameter = self.gru(x)
        # out -->[num_node,Batch, hidden_size=2*feature]
        out = self.drop(self.outer(out))# 降低维度
        #print(f"the hidden_parameter {hidden_parameter.shape}")
        return out,hidden_parameter
class CAGU(nn.Module):

    def __init__(self, n_channels, n_classes, bilinear=True):
        super(CAGU, self).__init__()

        self.index = 0
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear


        self.inc = DoubleConv(n_channels, 200)
        self.down1 = Down(200, 300)
        self.down2 = Down(300, 400)
        self.down3 = Down(400, 512)
        factor = 1


        self.up2 = Up(912, 400 // factor, bilinear)
        self.up3 = Up(700, 300 // factor, bilinear)
        self.up4 = Up(500, 100, bilinear)
        self.outc = OutConv(100, n_classes)

        # input_size,hidden_size,num_layers

        self.gru_model = GRU(625,625*2,1)
        self.conv1 = ChebConv(625, int(625*1.5), K=2)
        self.conv2 = ChebConv(int(625*1.5), 625, K=2)
        self.act = nn.ReLU()
        self.reduce_conv = nn.Conv2d(in_channels=512,out_channels=256,kernel_size=1) # reduce channel by 1x1 kernel
        self.expand_conv = nn.Conv2d(in_channels=256,out_channels=512,kernel_size=1)# expand channel by 1x1 kernel
    @staticmethod
    def get_adjancent_matrix_mate(graph_data,attention):
        '''
        graph_data: data :[B,1024,68,44]
        x_sort : [B,1024,1,1]
        '''

        A = np.zeros([graph_data.size(0), int(graph_data.size(1)), int(graph_data.size(1))])
        batch = graph_data.size(0)
        num_adjancent = graph_data.size(1)

        for k in range(batch):  # loop all batch data
            # A = np.zeros([int(self.train.size(1)), int(self.train.size(1))])
            train = graph_data[k, :, :, :]
            train = train.reshape(train.size(0), -1)
            for i in range(train.shape[0]):  # loop 512 channel image to build adjancent matrix
                for j in range(i + 1, train.shape[0]):
                    calculate_distance = abs(train[i, :] - train[j, :]).sum()
                    attention_distance = abs(attention[k,i,0,0]-attention[k,j,0,0])

                    if calculate_distance <= 180. and attention_distance<=0.05:

                        A[k, i, j] = 1.
                        A[k, j,i] = 1.
        return torch.from_numpy(A).to(torch.float32)  # [B, num_node,num_node]
    @staticmethod
    def get_adjancent_matrix(graph_data,x_sort):
        '''
        graph_data: data :[B,512,68,44]
        '''

        A = np.zeros([graph_data.size(0), int(graph_data.size(1)), int(graph_data.size(1))])
        batch = graph_data.size(0)
        num_adjancent = graph_data.size(1)
        for k in range(batch):  # loop all batch data
            train = graph_data[k, :, :, :]
            train = train.reshape(train.size(0), -1)
            for i in range(num_adjancent):  # loop 512 channel image to build adjancent matrix
                for j in range(i + 1, num_adjancent):
                    calculate_distance = abs(train[i, :].numpy() - train[j, :].numpy()).sum()
                    if calculate_distance <= 1415.:
                        A[k, i, j] = 1.
                        A[k, j, i] = 1.
        return torch.from_numpy(A).to(torch.float32)  # [B, num_node,num_node]

    @staticmethod
    def process_graph(graph_data):  # [B,N,N]
        """

        :param graph_data:
        :return:
        """

        N = graph_data.size(1)

        matrix_i = torch.randn(graph_data.shape)

        for i in range(graph_data.size(0)):
            matrix_i[i, :, :] = torch.eye(N, dtype=graph_data.dtype, device=graph_data.device)

        matrix_i = matrix_i.to(graph_data.device)
        graph_data += matrix_i  # A~ [N, N]

        degree_matrix = torch.sum(graph_data, dim=-1, keepdim=False)  # [b,N]

        degree_matrix = degree_matrix.pow(-1)

        degree_matrix[degree_matrix == float("inf")] = 0.  # [N]

        temp_metrix = torch.randn(graph_data.shape).to(graph_data.device)
        for i in range(graph_data.size(0)):
            degree_matrix_temp = torch.diag(degree_matrix[i, :])  # [N, N]
            temp_metrix[i, :, :] = degree_matrix_temp
        return torch.bmm(temp_metrix, graph_data)  # D^(-1) * A = \hat(A)
    def forward(self, x,label):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x4_gru = x4.view(x4.size(1),x4.size(0),-1)

        gru_out,hidden_parameter= self.gru_model(x4_gru)
        gru_out = gru_out.view(x4.shape)

        x_sequence = F.adaptive_avg_pool2d(x4,1) # size([1, 1024, 1, 1])
        x_sequence = self.expand_conv(torch.sigmoid(self.reduce_conv(x_sequence)))
        x_important = torch.sigmoid(x_sequence)

        x4 = torch.sigmoid(x_sequence)*x4

        x_intermediate = x4.detach().clone()

        graph_data = UNet.get_adjancent_matrix_mate(x_intermediate,x_important)
        graph_data = UNet.process_graph(graph_data).to(x4.device)

        x_intermediate = x_intermediate.view(x4.size(0), x4.size(1), -1)  # [B, 512, 1024]
        flow_x = x_intermediate  # [B, N, len]
        output_1 = self.act(self.conv1(flow_x, graph_data))
        output_2 = self.act(self.conv2(output_1, graph_data))
        output = output_2.view(x4.shape)
        outputs = 0.5*output + 0.5*gru_out

        x = self.up2(outputs, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)

        return logits


