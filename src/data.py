"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""

import torch
import os
import numpy as np
from PIL import Image
import cv2
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.utils.data_classes import Box
from glob import glob

from .tools import get_lidar_data, img_transform, normalize_img, gen_dx_bx

from torch.utils.data import Dataset, DataLoader

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
            img = Image.open(path).convert('RGB')
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

class KatiwjkData(Dataset):
    def __init__(self, root_path, input_img_size,  bsz):
        self.path = root_path
        self.img_size = input_img_size
        self.batch_size = bsz
        self.input_path = self.path
        self.get_input_paths()
        print(self.input_img_paths)

    def __getitem__(self, index):
        i = index * self.batch_size
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]

        # 3 Dimensional input images
        x = np.zeros((self.batch_size,)+ (3,) + (self.img_size[1],self.img_size[0]),  dtype="float32")
        for j, path in enumerate(batch_input_img_paths):
            img = cv2.imread(self.input_img_paths[index])
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_rgb = cv2.resize(img_rgb, self.img_size)
            img_rgb = normalize_img(img_rgb)
            # img_rgb  = np.einsum('kli->kil', img_rgb)
            print(img_rgb.shape)
            x[j] = img_rgb
        x = np.expand_dims(x, 1)
        return torch.from_numpy(x)

    def __len__(self):
        return len(self.input_img_paths)
    
    def get_input_paths(self):
        self.input_img_paths = sorted([
        os.path.join(self.input_path, fname)
        for fname in os.listdir(self.input_path)
    ])

class NuscData(torch.utils.data.Dataset):
    def __init__(self, nusc, is_train, data_aug_conf, grid_conf):
        self.nusc = nusc
        self.is_train = is_train
        self.data_aug_conf = data_aug_conf
        self.grid_conf = grid_conf

        self.scenes = self.get_scenes()
        self.ixes = self.prepro()

        dx, bx, nx = gen_dx_bx(grid_conf['xbound'], grid_conf['ybound'], grid_conf['zbound'])
        self.dx, self.bx, self.nx = dx.numpy(), bx.numpy(), nx.numpy()

        self.fix_nuscenes_formatting()

        print(self)

    def fix_nuscenes_formatting(self):
        """If nuscenes is stored with trainval/1 trainval/2 ... structure, adjust the file paths
        stored in the nuScenes object.
        """
        # check if default file paths work
        rec = self.ixes[0]
        sampimg = self.nusc.get('sample_data', rec['data']['CAM_FRONT'])
        imgname = os.path.join(self.nusc.dataroot, sampimg['filename'])

        def find_name(f):
            d, fi = os.path.split(f)
            d, di = os.path.split(d)
            d, d0 = os.path.split(d)
            d, d1 = os.path.split(d)
            d, d2 = os.path.split(d)
            return di, fi, f'{d2}/{d1}/{d0}/{di}/{fi}'

        # adjust the image paths if needed
        if not os.path.isfile(imgname):
            print('adjusting nuscenes file paths')
            fs = glob(os.path.join(self.nusc.dataroot, 'samples/*/samples/CAM*/*.jpg'))
            fs += glob(os.path.join(self.nusc.dataroot, 'samples/*/samples/LIDAR_TOP/*.pcd.bin'))
            info = {}
            for f in fs:
                di, fi, fname = find_name(f)
                info[f'samples/{di}/{fi}'] = fname
            fs = glob(os.path.join(self.nusc.dataroot, 'sweeps/*/sweeps/LIDAR_TOP/*.pcd.bin'))
            for f in fs:
                di, fi, fname = find_name(f)
                info[f'sweeps/{di}/{fi}'] = fname
            for rec in self.nusc.sample_data:
                if rec['channel'] == 'LIDAR_TOP' or (rec['is_key_frame'] and rec['channel'] in self.data_aug_conf['cams']):
                    rec['filename'] = info[rec['filename']]

    
    def get_scenes(self):
        # filter by scene split
        split = {
            'v1.0-trainval': {True: 'train', False: 'val'},
            'v1.0-mini': {True: 'mini_train', False: 'mini_val'},
        }[self.nusc.version][self.is_train]

        scenes = create_splits_scenes()[split]

        return scenes

    def prepro(self):
        samples = [samp for samp in self.nusc.sample]

        # remove samples that aren't in this split
        samples = [samp for samp in samples if
                   self.nusc.get('scene', samp['scene_token'])['name'] in self.scenes]

        # sort by scene, timestamp (only to make chronological viz easier)
        samples.sort(key=lambda x: (x['scene_token'], x['timestamp']))

        return samples
    
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

    def get_image_data(self, rec, cams):
        imgs = []
        rots = []
        trans = []
        intrins = []
        post_rots = []
        post_trans = []
        for cam in cams:
            samp = self.nusc.get('sample_data', rec['data'][cam])
            imgname = os.path.join(self.nusc.dataroot, samp['filename'])
            img = Image.open(imgname)
            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)

            sens = self.nusc.get('calibrated_sensor', samp['calibrated_sensor_token'])
            intrin = torch.Tensor(sens['camera_intrinsic'])
            rot = torch.Tensor(Quaternion(sens['rotation']).rotation_matrix)
            tran = torch.Tensor(sens['translation'])

            # augmentation (resize, crop, horizontal flip, rotate)
            resize, resize_dims, crop, flip, rotate = self.sample_augmentation()
            img, post_rot2, post_tran2 = img_transform(img, post_rot, post_tran,
                                                     resize=resize,
                                                     resize_dims=resize_dims,
                                                     crop=crop,
                                                     flip=flip,
                                                     rotate=rotate,
                                                     )
            
            # for convenience, make augmentation matrices 3x3
            post_tran = torch.zeros(3)
            post_rot = torch.eye(3)
            post_tran[:2] = post_tran2
            post_rot[:2, :2] = post_rot2

            imgs.append(normalize_img(img))
            intrins.append(intrin)
            rots.append(rot)
            trans.append(tran)
            post_rots.append(post_rot)
            post_trans.append(post_tran)

        return (torch.stack(imgs), torch.stack(rots), torch.stack(trans),
                torch.stack(intrins), torch.stack(post_rots), torch.stack(post_trans))

    def get_lidar_data(self, rec, nsweeps):
        pts = get_lidar_data(self.nusc, rec,
                       nsweeps=nsweeps, min_distance=2.2)
        return torch.Tensor(pts)[:3]  # x,y,z

    def get_binimg(self, rec):
        egopose = self.nusc.get('ego_pose',
                                self.nusc.get('sample_data', rec['data']['LIDAR_TOP'])['ego_pose_token'])
        trans = -np.array(egopose['translation'])
        rot = Quaternion(egopose['rotation']).inverse
        img = np.zeros((int(self.nx[0]), int(self.nx[1])))
        for tok in rec['anns']:
            inst = self.nusc.get('sample_annotation', tok)
            # add category for lyft
            if not inst['category_name'].split('.')[0] == 'vehicle':
                continue
            box = Box(inst['translation'], inst['size'], Quaternion(inst['rotation']))
            box.translate(trans)
            box.rotate(rot)

            pts = box.bottom_corners()[:2].T
            pts = np.round(
                (pts - self.bx[:2] + self.dx[:2]/2.) / self.dx[:2]
                ).astype(np.int32)
            pts[:, [1, 0]] = pts[:, [0, 1]]
            cv2.fillPoly(img, [pts], 1.0)

        return torch.Tensor(img).unsqueeze(0)

    def choose_cams(self):
        if self.is_train and self.data_aug_conf['Ncams'] < len(self.data_aug_conf['cams']):
            cams = np.random.choice(self.data_aug_conf['cams'], self.data_aug_conf['Ncams'],
                                    replace=False)
        else:
            cams = self.data_aug_conf['cams']
        return cams

    def __str__(self):
        return f"""NuscData: {len(self)} samples. Split: {"train" if self.is_train else "val"}.
                   Augmentation Conf: {self.data_aug_conf}"""

    def __len__(self):
        return len(self.ixes)


class VizData(NuscData):
    def __init__(self, *args, **kwargs):
        super(VizData, self).__init__(*args, **kwargs)
    
    def __getitem__(self, index):
        rec = self.ixes[index]
        
        cams = self.choose_cams()
        imgs, rots, trans, intrins, post_rots, post_trans = self.get_image_data(rec, cams)
        lidar_data = self.get_lidar_data(rec, nsweeps=3)
        binimg = self.get_binimg(rec)
        
        return imgs, rots, trans, intrins, post_rots, post_trans, lidar_data, binimg


class SegmentationData(NuscData):
    def __init__(self, *args, **kwargs):
        super(SegmentationData, self).__init__(*args, **kwargs)
    
    def __getitem__(self, index):
        rec = self.ixes[index]

        cams = self.choose_cams()
        imgs, rots, trans, intrins, post_rots, post_trans = self.get_image_data(rec, cams)
        binimg = self.get_binimg(rec)
        
        return imgs, rots, trans, intrins, post_rots, post_trans, binimg


def worker_rnd_init(x):
    np.random.seed(13 + x)


def compile_data(version, dataroot, data_aug_conf, grid_conf, bsz,
                 nworkers, parser_name):
    nusc = NuScenes(version='v1.0-{}'.format(version),
                    dataroot=os.path.join(dataroot, version),
                    verbose=False)
    parser = {
        'vizdata': VizData,
        'segmentationdata': SegmentationData,
    }[parser_name]
    traindata = parser(nusc, is_train=True, data_aug_conf=data_aug_conf,
                         grid_conf=grid_conf)
    valdata = parser(nusc, is_train=False, data_aug_conf=data_aug_conf,
                       grid_conf=grid_conf)

    trainloader = torch.utils.data.DataLoader(traindata, batch_size=bsz,
                                              shuffle=True,
                                              num_workers=nworkers,
                                              drop_last=True,
                                              worker_init_fn=worker_rnd_init)
    valloader = torch.utils.data.DataLoader(valdata, batch_size=bsz,
                                            shuffle=False,
                                            num_workers=nworkers)

    return trainloader, valloader


    
