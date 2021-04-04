#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File  : load_image.py
@Author: Moule Lin
@Date  : 2020/10/1 12:08

'''
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms
import os
from PIL import Image
import warnings
import cv2 as cv
import random
import scipy.io as sio
import numpy as np
import torch
import os
# warnings.filterwarnings("ignore")
trans_image = transforms.Compose([

    transforms.ToTensor(),

])
trans_label = transforms.Compose([

    transforms.ToTensor(),

])

class getImage(Dataset):
    def __init__(self,root):
        self.data_all = os.listdir(root) #
        self.root = root




    def __len__(self):
        return len(self.data_all)

    def __getitem__(self, index):

        image = self.data_all[index] #

        data_name = os.path.join(self.root,image)
        print(data_name)
        data = sio.loadmat(data_name)
        train = data["train"]
        label = data["label"]
        print(train.shape)

        data_train = getImage.to_tensor(train)
        data_train = data_train.permute(2,0,1)
        data_label = getImage.to_tensor_label(label)
        return data_train,data_label,image
    @staticmethod
    def to_tensor(data):
        return torch.from_numpy(data).to(torch.float)
    def to_tensor_label(data):
        return torch.from_numpy(data).to(torch.long)
    @staticmethod
    def pre_process_data(data, norm_dim):
        """
        :param data: np.array, original  data without normalization.
        :param norm_dim: int, normalization dimension. we use 2 dim(channel) to normalize
        :return:
            norm_base: list, [max_data, min_data], data of normalization base.
            norm_data: np.array, normalized traffic data.
        """
        norm_base = getImage.normalize_base(data, norm_dim)  # find the normalize base
        norm_data = getImage.normalize_data(norm_base[0], norm_base[1], data)  # normalize data

        return norm_base, norm_data

    @staticmethod
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
    @staticmethod
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
        normalized_data = (data - mid) / base

        return normalized_data

    @staticmethod
    def recover_data(max_data, min_data, data):
        """
        :param max_data: np.array, max data.
        :param min_data: np.array, min data.
        :param data: np.array, normalized data.
        :return:
            recovered_data: np.array, recovered data.
        """
        mid = min_data
        base = max_data - min_data

        recovered_data = data * base + mid

        return recovered_data
# pickle
def build_dataset_way2(mat, gt, ignored_labels=None):
    """Create a list of training samples based on an image and a mask.
    Args:
        mat: 3D hyperspectral matrix to extract the spectrums from      # 用来提取光谱的高光谱矩阵
        gt: 2D ground truth
        ignored_labels (optional): list of classes to ignore, e.g. 0 to remove
        unlabeled pixels
        return_indices (optional): bool set to True to return the indices of
        the chosen samples
    """

    samples = []
    labels = []
    # Check that image and ground truth have the same 2D dimensions
    assert mat.shape[:2] == gt.shape[:2]    # 检查维度是否相符,比如PaviaU的mat和gt都是(610, 340)
    indices_index = []
    index_num=0

    index_save = []
    flag = 0
    for label in np.unique(gt):
        indices = np.nonzero(gt == label)
       # 返回同一类标签的全部索引。（对gt每个元素判断是否为label，是的话为1否则为0，然后提取全部的非零元素的索引
       #  y0 = list(filter(lambda a: a != 0, indices[0]))
       #  y1 = list(filter(lambda a: a != 0, indices[1]))
        y0 = indices[0]
        y1 = indices[1]
        if flag == 0:
            random_int_ = np.random.randint(0,len(y0),int(len(y0)*0.1)+3) # 获得每个类的10%的数据
            flag+=1
        else:
            random_int_ = np.random.randint(0, len(y0), int(len(y0) * 0.1))
        z1 = [y0[i] for i in random_int_]
        z2 = [y1[i] for i in random_int_]
        for i in range(len(z1)):
            index_save.append([z1[i],z2[i]])
    np.random.shuffle(index_save)


    for i in index_save:
        samples.append(mat[i[0],i[1],:])
        labels.append(gt[i[0],i[1]])
    # print(np.asarray(samples).shape)
    return np.asarray(samples).reshape(186,169,176), np.asarray(labels).reshape(186,169)


