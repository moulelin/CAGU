# -*- coding: utf-8 -*-
"""
@File  : layers_gcn.py
@Author: Moule Lin
@Date  : 2020/10/1 12:08

"""
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
from torchvision import transforms

trans_image = transforms.Compose([
    transforms.ToTensor(),
   # transforms.Normalize((0.5), (0.5))

])
class GCN(nn.Module):
    def __init__(self, in_c, hid_c, out_c):
        super(GCN, self).__init__()
        self.linear_1 = nn.Linear(in_c, hid_c)
        self.linear_2 = nn.Linear(hid_c, out_c)
        self.act = nn.ReLU()

    def forward(self, graph_data,data,device):

        graph_data = GCN.process_graph(graph_data)

        flow_x = data.to(device)  # [B, N, len]
        graph_data = graph_data.to(device)
        output_1 = self.linear_1(flow_x)  # [B, N, hid_C]
        output_1 = self.act(torch.matmul(graph_data, output_1))  # [N, N], [B, N, Hid_C]
        output_2 = self.linear_2(output_1)
        output_2 = self.act(torch.matmul(graph_data, output_2))  # [B, N, 1, Out_C]

        return output_2.unsqueeze(2)

    @staticmethod
    def process_graph(graph_data): #  [B,N,N]
        N = graph_data.size(1)
        matrix_i = torch.randn(graph_data.shape)
        #matrix_i = torch.eye(N, dtype=graph_data.dtype, device=graph_data.device)
        for i in range(graph_data.size(0)):
            matrix_i[i,:,:] = torch.eye(N, dtype=graph_data.dtype, device=graph_data.device)
       # print(matrix_i.shape)
        graph_data += matrix_i  # A~ [N, N]
       # print(graph_data.shape)
        degree_matrix = torch.sum(graph_data, dim=-1, keepdim=False)  # [b,N]
       # print(degree_matrix.shape)
        degree_matrix = degree_matrix.pow(-1)
       # print(degree_matrix.shape)
        degree_matrix[degree_matrix == float("inf")] = 0.  # [N]
       # print(degree_matrix.shape)
        temp_metrix = torch.randn(graph_data.shape)

        for i in range(graph_data.size(0)):
            degree_matrix_temp = torch.diag(degree_matrix[i,:])  # [N, N]
            temp_metrix[i,:,:] = degree_matrix_temp
      #  print("jvghjfg",temp_metrix.shape)
        return torch.bmm(temp_metrix, graph_data)  # D^(-1) * A = \hat(A)
if __name__ == '__main__':
    data = torch.randn(4,10,10)
    a = GCN.process_graph(data)
    print(a)

class GCN_train(nn.Module):
    def __init__(self,train,label):
        super(GCN_train, self).__init__()
        self.train = train.detach().cpu() # [B,C,N,N]->[B 512 32 32]
        self.label = label.detach().cpu() # [B,C,N,N]->[B 512 32 32]


    def get_adjancent_matrix(self):
        '''
        x : data :[1,512,32,32]
        '''

        batch,num,row,col =  self.train.size(0),self.train.size(1),self.train.size(2),self.train.size(3)
        A = np.zeros([self.train.size(0),int(self.train.size(1)), int(self.train.size(1))])
        data_train_all = []
        data_label_all = []
        for k in range(batch): # loop all batch data
            # A = np.zeros([int(self.train.size(1)), int(self.train.size(1))])
            train = self.train[k,:,:,:].cpu().numpy()
            train_data = []
            label = self.label[k,:,:,:].cpu().numpy()
            label_data = []
            for i in range(num):# loop 512 channel image to build adjancent matrix
                data_temp = train[i,:,:].reshape(1,-1)
                data_temp_label = label[i,:,:].reshape(1,-1)
                train_data.append(np.squeeze(data_temp))
                label_data.append(np.squeeze(data_temp_label))
                for j in range(i+1,num):
                    data_temp_other = np.squeeze(train[j,:,:].reshape(1,-1))
                    calculate_distance = abs(data_temp-data_temp_other).sum()
                    if calculate_distance<=500.:
                        A[k,i,j] = 1.
                        A[k, j, i] = 1.
            data_train_all.append(np.array(train_data))
            data_label_all.append(np.array(label_data))
            # print(np.array(data_label_all).shape, "asdfasdf")
        return A,np.array(data_train_all),np.array(data_label_all)

    def train_gcn(self,matrix,train_data,label_data):
        '''
        matrix:[B,N N]->[batch,512,512]
        train_data:[B, N lens]->[batch, 512,1024]
        '''
        in_c = len(train_data[0,0,:]) # 1024

        matrix = torch.from_numpy(matrix[0,:,:]).float()
        train_data = torch.from_numpy(train_data).float()
        label_data = torch.from_numpy(label_data).float()
        # print(train_data.shape)
        batch_size = train_data.size(0) # get batch size
        my_net = GCN(in_c,int(in_c*1.5),in_c)

        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

        my_net = my_net.to(device)

        criterion = nn.MSELoss()

        optimizer = optim.Adam(params=my_net.parameters())

        # Train model
        Epoch = 11

        my_net.train()
        after_train_train_data = []
        after_train_label_data = []
        for batch in range(batch_size): # loop all batch data
            for epoch in range(Epoch):
                epoch_loss = 0.0
                start_time = time.time()
                train_temp = train_data[batch,:,:]
                label_temp = label_data[batch,:,:]
                my_net.zero_grad()
                predict_value=my_net(matrix, train_temp, device).to(torch.device("cpu"))
                predict_value = predict_value.squeeze(2)

                loss = criterion(predict_value, label_temp)
                epoch_loss = loss.item()
                loss.backward()
                optimizer.step()
                end_time = time.time()
                # if epoch and epoch%10==0:
                #     print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                #     print("ChebNet loss---->Epoch: {:d}, Loss: {:02.4f}, Time: {:02.2f} mins".format(epoch,  epoch_loss ,(end_time-start_time)/60))
                #     print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                if epoch==Epoch-1:
                    after_train_train_data.append(predict_value.view(512,32,32).detach().numpy())
        return torch.from_numpy(np.array(after_train_train_data))



