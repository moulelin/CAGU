# -*- coding: utf-8 -*-
'''
@File  : extra_houston_10%.py
@Author: Moule Lin
@Date  : 2020/12/6 11:23
@Github: https://github.com/mou
'''
import scipy.io as sio

from collections import Counter
import numpy as np
from random import randint
import pickle
import queue
import pandas as pd
next_step=[ #Eight direction
[-1,1],
[-1, 0],
[1,1],#
[0, -1],
[0, 1],
[-1, -1],
[1, 0],
[1,-1],
]
w_sample_ = 200
h_sample_ = 200
def sampling_group_class(ground_truth,data,ignore_class=None):
    """

    :param ground_truth: image label
    :param data:
    :param ignore_class: e.g. 0
    :return:
    group each class and divide into
    """

    w,h,c = data.shape
    w_random = randint(200,w-200)
    h_random = randint(100,h-100)
    point_FIFO = queue.Queue() # 新建一个队列，类似广度优先搜索
    w_sample = 200
    h_sample = 200
    sample_mat = np.zeros((w_sample, h_sample,c), dtype=np.float)

    visit = np.zeros((w,h), dtype=np.int) # 标记数组，标记是否已经入队列了，
    point_FIFO.put((w_random,h_random)) # 先加入中心点
    visit[w_random][h_random] = 1 # visit数组标记

    while(True): # 重新找到随机点中心
        temp_index = point_FIFO.get()
        if ground_truth[temp_index[0]][temp_index[1]] == 0.: # 重新找到新的中心点
            for i in next_step: # 八个方向
                if visit[temp_index[0]+i[0]][temp_index[1]+i[1]] == 0:
                    point_FIFO.put((temp_index[0]+i[0],temp_index[1]+i[1]))
                    visit[temp_index[0]+i[0]][temp_index[1]+i[1]] = 1
        else:
            center = temp_index
            break
    return center
def pre_process_data(data, norm_dim):
    """
    :param data: np.array, original  data without normalization.
    :param norm_dim: int, normalization dimension. we use 2 dim(channel) to normalize
    :return:
        norm_base: list, [max_data, min_data], data of normalization base.
        norm_data: np.array, normalized traffic data.
    """
    norm_base = normalize_base(data, norm_dim)  # find the normalize base
    norm_data = normalize_data(norm_base[0], norm_base[1], data)  # normalize data

    return norm_base, norm_data


def normalize_base(data, norm_dim):
    """
    :param data: np.array, original  data without normalization.
    :param norm_dim: int, normalization dimension.
    :return:
        max_data: np.array
        min_data: np.array
    """

    max_data = np.max(data, norm_dim, keepdims=True)  # [N, T, D] , norm_dim=1, [N, 1, D]
    min_data = np.min(data, norm_dim, keepdims=True)

    return max_data, min_data

def normalize_data(max_data, min_data, data):
    """
    :param max_data: np.array, max data.
    :param min_data: np.array, min data.
    :param data: np.array, original  data without normalization.
    :return:
        np.array, normalized traffic data.
    """
    mid = min_data
    base = max_data - min_data
    normalized_data = (data - mid) / (base+0.01)

    return normalized_data

def sixth_point_build_dataset():

    data = sio.loadmat("Houston_2018.mat")["Houston_2018"]
    ground_truth = sio.loadmat("Houston_gt_2018.mat")["Houston_gt_2018"]
    w,h,c = data.shape
    sample_mat = np.zeros((w_sample_,w_sample_,c),dtype=np.float32)
    sample_gt = np.zeros((w_sample_,w_sample_),dtype=np.int)
    for epoch in range(1,41):
        for i in range(8): # 64 group
            for j in range(8):
                center = sampling_group_class(ground_truth,data)
                temp_data = ground_truth[center[0]-12:center[0]+13,center[1]-12:center[1]+13]
                statistics = len(temp_data[temp_data!=0.])
                # print("++++++++++++++++++")
                # print(statistics)
                # print("++++++++++++++++++")
                if statistics>=220:
                    sample_mat[i*25:(i+1)*25,j*25:(j+1)*25,:] = data[center[0]-12:center[0]+13,center[1]-12:center[1]+13,:]
                    sample_gt[i * 25:(i + 1) * 25, j * 25:(j + 1) * 25] = ground_truth[center[0] - 12:center[0] + 13,
                                                                              center[1] - 12:center[1] + 13]
                else:
                    while(True):
                        center = sampling_group_class(ground_truth, data)
                        temp_data = ground_truth[center[0] - 12:center[0] + 13, center[1] - 12:center[1] + 13]
                        statistics = len(temp_data[temp_data != 0.])
                        if statistics >= 220:
                            sample_mat[i * 25:(i + 1) * 25, j * 25:(j + 1) * 25, :] = data[center[0] - 12:center[0] + 13,center[1] - 12:center[1] + 13,:]
                            sample_gt[i * 25:(i + 1) * 25, j * 25:(j + 1) * 25] = ground_truth[center[0] - 12:center[0] + 13,center[1] - 12:center[1] + 13]
                            break
        _, sample_mat = pre_process_data(sample_mat, -1)
        sio.savemat(f"../dataset/Houston_2018/sample_houston_{str(epoch)}.mat",{"train":sample_mat,"label":sample_gt},do_compression = True)
        print(epoch)




if __name__ == '__main__':
    sixth_point_build_dataset()


