#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File  : layers_chebnet.py
@Author: Moule Lin
@Date  : 2020/8/29 12:08

'''
import torch
import torch.nn as nn
import torch.nn.init as init


class ChebConv(nn.Module):
    """
    The ChebNet convolution operation.

    :param in_c: int, number of input channels.
    :param out_c: int, number of output channels.
    :param K: int, the order of Chebyshev Polynomial.
    """
    def __init__(self, in_c, out_c, K, bias=True, normalize=True):
        super(ChebConv, self).__init__()
        self.normalize = normalize

        self.weight = nn.Parameter(torch.Tensor(K + 1, 1, in_c, out_c))  # [K+1, 1, in_c, out_c]
        init.xavier_normal_(self.weight)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, 1, out_c))
            init.zeros_(self.bias)
        else:
            self.register_parameter("bias", None)

        self.K = K + 1

    def forward(self, inputs, graph):
        """
        :param inputs: the input data, [B, N, C]
        :param graph: the graph structure, [N, N]
        :return: convolution result, [B, N, D]
        """
        L = ChebConv.get_laplacian(graph, self.normalize)  # [N, N]
        mul_L = self.cheb_polynomial(L).to(graph.device)  # [K, 1, N, N]
      #  print("mul_L shape is",mul_L.shape)
        #   print("at parts shape of input and graph",inputs.shape,graph.shape)
        result = torch.matmul(mul_L, inputs)  # [K, B, N, C]

        result = torch.matmul(result, self.weight)  # [K, B, N, D]
        result = torch.sum(result, dim=0) + self.bias  # [B, N, D]

        return result

    def cheb_polynomial(self, laplacian):
        """
        Compute the Chebyshev Polynomial, according to the graph laplacian.

        :param laplacian: the graph laplacian, [N, N].
        :return: the multi order Chebyshev laplacian, [K, N, N].
        """
        N = laplacian.size(1)  # [N, N]
        B = laplacian.size(0)
        multi_order_laplacian = torch.zeros([self.K, B,N, N], device=laplacian.device, dtype=torch.float)  # [K, B,N, N]
        for i in range(B):
            multi_order_laplacian[0,i,:,:] = torch.eye(N, device=laplacian.device, dtype=torch.float)

            if self.K == 1:
                return multi_order_laplacian
            else:
                multi_order_laplacian[1,i,:,:] = laplacian[i,:,:]
                if self.K == 2:
                    return multi_order_laplacian
                else:
                    for k in range(2, self.K):
                        multi_order_laplacian[k,i,:,:] = 2 * torch.mm(laplacian[i,:,:], multi_order_laplacian[k-1,i,:,:]) - \
                                                   multi_order_laplacian[k-2,i,:,:]

        return multi_order_laplacian

    @staticmethod
    def get_laplacian(graph, normalize):
        """
        return the laplacian of the graph.

        :param graph: the graph structure without self loop, [B, N, N].
        :param normalize: whether to used the normalized laplacian.
        :return: graph laplacian.
        """
        if normalize:
            L = torch.randn(graph.shape)
            for i in range(graph.size(0)):
                D = torch.diag(torch.sum(graph[i,:,:], dim=-1) ** (-1 / 2))
                X = torch.eye(graph.size(1), device=graph.device, dtype=graph.dtype) - torch.mm(torch.mm(D, graph[i,:,:]), D)
                L[i,:,:] = X
        else:
            D = torch.diag(torch.sum(graph, dim=-1))
            L = D - graph
        return L


class ChebNet(nn.Module):
    def __init__(self, in_c, hid_c, out_c, K):
        """
        :param in_c: int, number of input channels.
        :param hid_c: int, number of hidden channels.
        :param out_c: int, number of output channels.
        :param K:
        """
        super(ChebNet, self).__init__()
        self.conv1 = ChebConv(in_c=in_c, out_c=hid_c, K=K)
        self.conv2 = ChebConv(in_c=hid_c, out_c=out_c, K=K)
        self.act = nn.ReLU()

    def forward(self, graph,data, device):
        '''
        data ： [B, N,len] ->[1,1200,len]
        graph:[N, N]->[1200,1200]

        '''
        flow_x = data # [B, N, len]
        output_1 = self.act(self.conv1(flow_x, graph))
        output_2 = self.act(self.conv2(output_1, graph))

        return output_2.unsqueeze(2)


