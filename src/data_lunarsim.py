import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import cv2
import os
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from .tools import get_lidar_data, img_transform, normalize_img, gen_dx_bx
from .tools import (ego_to_cam, get_only_in_img_mask, denormalize_img,
                    SimpleLoss, get_val_info, add_ego, gen_dx_bx,
                    get_nusc_maps, plot_nusc_map)
# from src.tools import DiceLoss
from .tools import get_batch_iou
from .models import compile_model
from tensorboardX import SummaryWriter
from time import time
import gc
import pkbar

class LunarsimData(Dataset):
    def __init__(self, target_paths, input_paths, input_img_size, bsz, num_cams):
        self.img_size = input_img_size
        self.batch_size = bsz
        self.num_cams = num_cams
        self.input_paths = input_paths
        self.target_size = (200, 200) 
        self.target_paths = target_paths
        self.intrins = self.get_intrins()
        self.rots = self.get_rots()
        self.trans = self.get_trans()

    # Augumentation function adopted from LSS repo
    def sample_augmentation(self):
        H, W = self.data_aug_conf['H'], self.data_aug_conf['W']
        fH, fW = self.data_aug_conf['final_dim']
        if self.is_train:
            resize = np.random.uniform(*self.data_aug_conf['resize_lim'])
            resize_dims = (int(W*resize), int(H*resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*self.data_aug_conf['bot_pct_lim']))*newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.data_aug_conf['rand_flip'] and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.data_aug_conf['rot_lim'])
        else:
            resize = max(fH/H, fW/W)
            resize_dims = (int(W*resize), int(H*resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.data_aug_conf['bot_pct_lim']))*newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0
        return resize, resize_dims, crop, flip, rotate

    def __getitem__(self, index):
        # load all cameras so the size of tensor is [4, 6, 3, height, width]
        #                                           bsz, numcams, dim
        x = np.zeros((self.num_cams,) + (3,) + 
        (self.img_size[1], self.img_size[0]), dtype="float32")
        for cam_index, path in enumerate(self.input_paths[index]):
            img = cv2.imread(path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_rgb = cv2.resize(img_rgb, self.img_size)
            img_rgb = normalize_img(img_rgb)
            x[cam_index] = img_rgb

        # Load the target images so the shape is (bsz, 1, height, width)
        y = np.zeros((1,) + self.target_size, dtype="float32")

        y = self.load_ground(self.target_paths[index])

        # There is no augmentation in data loader
        # so these two tensors are empty or identity
        post_rot = self.get_post_rots()
        post_trans = self.get_post_trans()
        
        return  (torch.from_numpy(x).float(), torch.from_numpy(self.rots).float(), 
                  torch.from_numpy(self.trans).float(), torch.from_numpy(self.intrins).float(),
                  torch.from_numpy(post_rot).float(), torch.from_numpy(post_trans).float(), torch.from_numpy(y).float())

    def __len__(self):
        return len(self.target_paths)
            
    def load_ground(self, path):
        # Read and resize image
        ground_img = cv2.imread(path)
        ground_img = cv2.resize(ground_img, self.target_size)
        # The segmentation masks are color coded
        # only usefull information about rocks is 
        # in the green channel
        (R, G, B) = cv2.split(ground_img)
        target = np.expand_dims(G, 2)
        # change the order of channels 
        # when loading into model
        target  = np.einsum('kli->ikl', target)
        # clip the target to range 0 - 1
        target = np.clip(target, 0, 1)
        return target
    
    def get_intrins(self):
        # Taken from unity
        # the intrisic matrix defines how
        # camera points translate to real world points 
        
        matrix = [[0.788690269, 0.0, 0.0], 
                  [0.0, -1.39473653, 0.0],
                  [0.0, 0.0, -1.01207244]]
        # make this into size (bsz, cams, 3, 3)
        intrs = np.zeros((self.num_cams,) + (3,3),  dtype="float32")
        for i in range(self.num_cams):
            intrs[i] = matrix
        
        return np.array(intrs)
    
    def get_rots(self):
        # rotation matrix for every camera in respect to ego frame
        rots = []
        forward_left_cam = [[-0.7543766, -0.1333224, -0.6427605],
                        [0.0954112, -0.9910294,  0.0935814],
                        [-0.6494710,  0.0092690,  0.7603298]]
        forward_cam = [[-0.9847720, -0.1738506,  0.0000353],
                   [0.1738506, -0.9847720,  0.0002230],
                   [-0.0000040,  0.0002257,  1.0000000]]
        forward_right_cam = [[-0.7543818, -0.1330322,  0.6428145],
                         [0.1738506, -0.9847720,  0.0002230],
                         [0.6329961,  0.1119219,  0.7660218]]
        back_left_cam = [[0.3368156,  0.0592483, -0.9397047],
                     [0.0433705, -0.9979352, -0.0473746],
                     [-0.9405712, -0.0247990, -0.3386897]]
        back_cam = [[1.0000000,  0.0002055,  0.0000040],
                [0.0002055, -0.9999999,  0.0002257],
                [0.0000040, -0.0002257, -1.0000000]]
        back_right_cam = [[0.3368080,  0.0596725,  0.9396806],
                      [0.0433701, -0.9979137,  0.0478254],
                      [0.9405740,  0.0246461, -0.3386934]]
        rots.append(forward_left_cam)
        rots.append(forward_cam)
        rots.append(forward_right_cam)
        rots.append(back_left_cam)
        rots.append(back_cam)
        rots.append(back_right_cam)
        return np.array(rots)
    
    def get_post_rots(self):
        post_rot = np.zeros((self.num_cams,) + (3,3),  dtype="float32")
        for i in range(self.num_cams):
            post_rot[i] = torch.eye(3)
        return post_rot
    
    def get_post_trans(self):
        post_trans = np.zeros((self.num_cams,) + (3,),  dtype="float32")
        for i in range(self.num_cams):
            post_trans[i] = torch.zeros(3)
        return post_trans
        
    def get_trans(self):
        trans = []
        forward_cam = [-0.51, -0.775, 0.264]
        forward_right_cam = [0.305, -1.172, 0.319]
        forward_left_cam = [-1.32, -1.177, 0.289]
        back_left_cam = [-1.19, -1.36, -1.69]
        back_right_cam = [0.102, -1.438, -1.69]
        back_cam = [-0.622, -1.998, -2.02]
        trans.append(forward_left_cam)
        trans.append(forward_cam)
        trans.append(forward_right_cam)
        trans.append(back_left_cam)
        trans.append(back_cam)
        trans.append(back_left_cam)
        return np.array(trans)
        

class LunarsimDataAug(Dataset):
    def __init__(self, target_paths, input_paths, input_img_size, bsz, num_cams, data_aug_conf):
        self.img_size = input_img_size
        self.batch_size = bsz
        self.num_cams = num_cams
        self.input_paths = input_paths
        self.target_size = (200, 200) 
        self.target_paths = target_paths
        self.intrins = self.get_intrins()
        self.rots = self.get_rots()
        self.trans = self.get_trans()
        self.data_aug_conf = data_aug_conf
        self.is_train = True

    # Augumentation function adopted from LSS repo
    def sample_augmentation(self):
        H, W = self.data_aug_conf['H'], self.data_aug_conf['W']
        fH, fW = self.data_aug_conf['final_dim']
        if self.is_train:
            resize = np.random.uniform(*self.data_aug_conf['resize_lim'])
            resize_dims = (int(W*resize), int(H*resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*self.data_aug_conf['bot_pct_lim']))*newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.data_aug_conf['rand_flip'] and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.data_aug_conf['rot_lim'])
        else:
            resize = max(fH/H, fW/W)
            resize_dims = (int(W*resize), int(H*resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.data_aug_conf['bot_pct_lim']))*newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0
        return resize, resize_dims, crop, flip, rotate

    def __getitem__(self, index):
        # load all cameras so the size of tensor is [4, 6, 3, height, width]
        #                                           bsz, numcams, dim
        x = np.zeros((self.num_cams,) + (3,) + 
        (self.img_size[1], self.img_size[0]), dtype="float32")
        post_rots = []
        post_trans = []
        for cam_index, path in enumerate(self.input_paths[index]):
            try:
                img = Image.open(path).convert('RGB')
            except:
                print(path)
            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)
            resize, resize_dims, crop, flip, rotate = self.sample_augmentation()
            img, post_rot2, post_tran2 = img_transform(img, post_rot, post_tran,
                                                     resize=resize,
                                                     resize_dims=resize_dims,
                                                     crop=crop,
                                                     flip=flip,
                                                     rotate=rotate,
                                                     )
            # img_rgb = cv2.resize(img_rgb, self.img_size)

            post_tran = torch.zeros(3)
            post_rot = torch.eye(3)
            post_tran[:2] = post_tran2
            post_rot[:2, :2] = post_rot2

            post_rots.append(post_rot)
            post_trans.append(post_tran)
            img_rgb = normalize_img(img)
            x[cam_index] = img_rgb

        # Load the target images so the shape is (bsz, 1, height, width)
        y = np.zeros((1,) + self.target_size, dtype="float32")

        y = self.load_ground(self.target_paths[index])

        # There is no augmentation in data loader
        # so these two tensors are empty or identity
        post_rots = self.get_post_rots()
        post_trans = self.get_post_trans()
        
        return  (torch.from_numpy(x).float(), torch.from_numpy(self.rots).float(), 
                  torch.from_numpy(self.trans).float(), torch.from_numpy(self.intrins).float(),
                  torch.from_numpy(post_rots).float(), torch.from_numpy(post_trans).float(), torch.from_numpy(y).float())
        # return  (torch.from_numpy(x).float(), torch.from_numpy(self.rots).float(), 
        #           torch.from_numpy(self.trans).float(), torch.from_numpy(self.intrins).float(),
        #       torch.stack(post_rots), torch.stack(post_trans).float(), torch.from_numpy(y).float())

    def __len__(self):
        return len(self.target_paths)
            
    def load_ground(self, path):
        # Read and resize image
        ground_img = cv2.imread(path)
        ground_img = cv2.resize(ground_img, self.target_size)
        # The segmentation masks are color coded
        # only usefull information about rocks is 
        # in the green channel
        (R, G, B) = cv2.split(ground_img)
        target = np.expand_dims(G, 2)
        # change the order of channels 
        # when loading into model
        target  = np.einsum('kli->ikl', target)
        # clip the target to range 0 - 1
        target = np.clip(target, 0, 1)
        return target
    
    def get_intrins(self):
        # Taken from unity
        # the intrisic matrix defines how
        # camera points translate to real world points 
        
        matrix = [[0.788690269, 0.0, 0.0], 
                  [0.0, -1.39473653, 0.0],
                  [0.0, 0.0, -1.01207244]]
        # make this into size (bsz, cams, 3, 3)
        intrs = np.zeros((self.num_cams,) + (3,3),  dtype="float32")
        for i in range(self.num_cams):
            intrs[i] = matrix
        
        return np.array(intrs)
    
    def get_rots(self):
        # rotation matrix for every camera in respect to ego frame
        rots = []
        forward_left_cam = [[-0.7543766, -0.1333224, -0.6427605],
                        [0.0954112, -0.9910294,  0.0935814],
                        [-0.6494710,  0.0092690,  0.7603298]]
        forward_cam = [[-0.9847720, -0.1738506,  0.0000353],
                   [0.1738506, -0.9847720,  0.0002230],
                   [-0.0000040,  0.0002257,  1.0000000]]
        forward_right_cam = [[-0.7543818, -0.1330322,  0.6428145],
                         [0.1738506, -0.9847720,  0.0002230],
                         [0.6329961,  0.1119219,  0.7660218]]
        back_left_cam = [[0.3368156,  0.0592483, -0.9397047],
                     [0.0433705, -0.9979352, -0.0473746],
                     [-0.9405712, -0.0247990, -0.3386897]]
        back_cam = [[1.0000000,  0.0002055,  0.0000040],
                [0.0002055, -0.9999999,  0.0002257],
                [0.0000040, -0.0002257, -1.0000000]]
        back_right_cam = [[0.3368080,  0.0596725,  0.9396806],
                      [0.0433701, -0.9979137,  0.0478254],
                      [0.9405740,  0.0246461, -0.3386934]]
        rots.append(forward_left_cam)
        rots.append(forward_cam)
        rots.append(forward_right_cam)
        rots.append(back_left_cam)
        rots.append(back_cam)
        rots.append(back_right_cam)
        return np.array(rots)
    
    def get_post_rots(self):
        post_rot = np.zeros((self.num_cams,) + (3,3),  dtype="float32")
        for i in range(self.num_cams):
            post_rot[i] = torch.eye(3)
        return post_rot
    
    def get_post_trans(self):
        post_trans = np.zeros((self.num_cams,) + (3,),  dtype="float32")
        for i in range(self.num_cams):
            post_trans[i] = torch.zeros(3)
        return post_trans
        
    def get_trans(self):
        trans = []
        forward_cam = [-0.51, -0.775, 0.264]
        forward_right_cam = [0.305, -1.172, 0.319]
        forward_left_cam = [-1.32, -1.177, 0.289]
        back_left_cam = [-1.19, -1.36, -1.69]
        back_right_cam = [0.102, -1.438, -1.69]
        back_cam = [-0.622, -1.998, -2.02]
        trans.append(forward_left_cam)
        trans.append(forward_cam)
        trans.append(forward_right_cam)
        trans.append(back_left_cam)
        trans.append(back_cam)
        trans.append(back_left_cam)
        return np.array(trans)