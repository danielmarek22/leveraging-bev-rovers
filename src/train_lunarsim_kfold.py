import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import cv2
import os
from PIL import Image
from sklearn.model_selection import KFold

import torch
from torch.utils.data import Dataset, DataLoader
from .tools import (ego_to_cam, get_only_in_img_mask, denormalize_img,
                    SimpleLoss, get_val_info, add_ego, gen_dx_bx,
                    get_nusc_maps, plot_nusc_map)
# from src.tools import DiceLoss
from .tools import get_batch_iou
from .models import compile_model
from .data_lunarsim import LunarsimData, LunarsimDataAug
from tensorboardX import SummaryWriter
from time import time
import gc
import pkbar
import mlflow


def load_ground(path):
    # Read and resize image
    ground_img = cv2.imread(path)
    ground_img = cv2.resize(ground_img, (200, 200))
    (R, G, B) = cv2.split(ground_img)
    target = np.expand_dims(G, 2)

    return target

def get_camera_paths(data_dir):
    
    cams = [ 'forward_left', 'forward', 'forward_right',
            'back_left', 'back', 'back_right']
    input_img_paths = []
    cam_input_paths = []
    # get all the paths
    for cam in cams:
        cam_input_path = data_dir + 'camera_{}'.format(cam)
        cam_input_paths = sorted([
        os.path.join(cam_input_path, fname)
        for fname in os.listdir(cam_input_path)
        ])
        input_img_paths.append(cam_input_paths)
    
    return input_img_paths

def get_traverse_paths(data_dir, target_dir):   
    input_img_paths = get_camera_paths(data_dir)
    target_paths = sorted([
        os.path.join(target_dir, fname)
        for fname in os.listdir(target_dir)
        if fname.endswith(".png")
    ])
    # target_paths = paths[:min_len]
    input_img_paths = np.transpose(input_img_paths)
    return input_img_paths.tolist(), target_paths


class DiceLoss(torch.nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.nn.functional.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice

def viz_preds(preds, binimgs):
    preds = preds.to('cpu')
    preds = preds.detach().numpy()
    pred = preds[0]
    # preds = np.einsum('kli->lik', preds)
    pred = np.squeeze(pred)

    binimgs = binimgs.to('cpu')
    binimgs = binimgs.detach().numpy()
    binimg = binimgs[0]
    binimg = np.squeeze(binimg)
    # plt.imshow(pred)

    fig, (ax1, ax2) =  plt.subplots(1, 2, figsize=(10,10))
    ax1.imshow(pred)
    ax2.imshow(binimg)
    plt.show()

def train_cv(
            experiment_name = "LSS on Lunarsim data, dice loss, 40 epoches, mobilenet encoder", 
            nepochs=40,         #10000
            gpuid=0,
            H=900, W=1600,
            resize_lim=(0.193, 0.225),
            final_dim=(128, 352),
            bot_pct_lim=(0.0, 0.22),
            rot_lim=(-5.4, 5.4),
            rand_flip=True,
            ncams=6,
            max_grad_norm=5.0,
            pos_weight=2.13,
            logdir='./runs',

            xbound=[-50.0, 50.0, 0.5],
            ybound=[-50.0, 50.0, 0.5],
            zbound=[-10.0, 10.0, 20.0],
            dbound=[4.0, 45.0, 1.0],

            bsz=8,
            nworkers=10,
            lr=1e-3,
            weight_decay=1e-7,
            ):
    grid_conf = {
        'xbound': xbound,
        'ybound': ybound,
        'zbound': zbound,
        'dbound': dbound,
    }
    data_aug_conf = {
                    'resize_lim': resize_lim,
                    'final_dim': final_dim,
                    'rot_lim': rot_lim,
                    'H': H, 'W': W,
                    'rand_flip': rand_flip,
                    'bot_pct_lim': bot_pct_lim,
                    'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                             'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
                    'Ncams': ncams,
                }

    img_size = (352, 128)
    batch_size = 8
    num_cameras = 6

    dir_path = os.path.dirname(os.path.realpath(__file__))
    print(dir_path)

    print(experiment_name)
    mlflow.set_tracking_uri("http://misc.mlflow.kplabs.pl")
    mlflow.set_experiment("BEVER LSS on LunarSim")

    # Define the list of traverse paths and target paths
    traverse_paths = [
        "./lunarsim_bev/traverse0/",
        "./lunarsim_bev/traverse1/",
        "./lunarsim_bev/traverse2/",
        "./lunarsim_bev/traverse3/",
        "./lunarsim_bev/traverse4/",
        "./lunarsim_bev/traverse5/",
        "./lunarsim_bev/traverse6/",
        "./lunarsim_bev/traverse7/"
    ]

    target_paths = [
        "./lunarsim_bev/traverse0/bev_seg/",
        "./lunarsim_bev/traverse1/bev_seg/",
        "./lunarsim_bev/traverse2/bev_seg/",
        "./lunarsim_bev/traverse3/bev_seg/",
        "./lunarsim_bev/traverse4/bev_seg/",
        "./lunarsim_bev/traverse5/bev_seg/",
        "./lunarsim_bev/traverse6/bev_seg/",
        "./lunarsim_bev/traverse7/bev_seg"
    ]

    # Combine traverse paths and target paths into tuples
    traverse_target_paths = list(zip(traverse_paths, target_paths))

    # Initialize KFold cross-validator
    kf = KFold(n_splits=len(traverse_paths))
    test_results_iou = []
    with mlflow.start_run(run_name=experiment_name):

        for train_index, test_index in kf.split(traverse_target_paths):
            print(train_index)
            print(test_index)
            # Split data into train and test folds
            train_folds = [traverse_target_paths[i] for i in train_index]
            test_fold = [traverse_target_paths[i] for i in test_index]

            # Extract paths for training and testing data from folds
            X_train = []
            y_train = []
            for traverse_path, target_path in train_folds:
                traverse_train_paths, traverse_train_target_paths = get_traverse_paths(traverse_path, target_path)
                X_train.extend(traverse_train_paths)
                y_train.extend(traverse_train_target_paths)

            X_test = []
            y_test = []
            for traverse_path, target_path in test_fold:
                traverse_test_paths, traverse_test_target_paths = get_traverse_paths(traverse_path, target_path)
                X_test.extend(traverse_test_paths)
                y_test.extend(traverse_test_target_paths)


            # Define the data and data loaders
            train_data = LunarsimDataAug(y_train, X_train, img_size, batch_size, num_cameras, data_aug_conf)
            # train_data = LunarsimData(y_train, X_train, img_size, batch_size, num_cameras)
            test_data = LunarsimData(y_test, X_test, img_size, batch_size, num_cameras)
            
            trainloader = DataLoader(train_data, batch_size=bsz, shuffle=True, num_workers=nworkers)
            testloader = DataLoader(test_data, batch_size=bsz, shuffle=True, num_workers=nworkers)

            gpuid=0
            device = torch.device('cpu') if gpuid < 0 else torch.device(f'cuda:{gpuid}')
            model = compile_model(grid_conf, data_aug_conf, outC=1)
            model.to(device)
            mlflow.pytorch.autolog()

            opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            # testing smaller learning rate
            # opt = torch.optim.Adam(model.parameters(), lr=lr)

            # loss_fn = SimpleLoss(pos_weight).cuda(gpuid)
            loss_fn = DiceLoss().cuda(gpuid)

            writer = SummaryWriter(logdir=logdir)
            val_step = 1000

            model.train()
            counter = 0
            gc.collect()
            val_loss_checkpoint = 100
            iou_loss_checkpoint = -1
            for epoch in range(nepochs):
                np.random.seed()
                # kbar = pkbar.Kbar(target=len(trainloader), epoch=epoch, num_epochs=nepochs, width=50, always_stateful=False)
                for batchi, (imgs, rots, trans, intrins, post_rots, post_trans, binimgs) in enumerate(trainloader):
                    t0 = time()
                    opt.zero_grad()
                    preds = model(imgs.to(device).float(),
                            rots.to(device).float(),
                            trans.to(device).float(),
                            intrins.to(device).float(),
                            post_rots.to(device),
                            post_trans.to(device),
                            )
                    # print(np.unique(binimgs))
                    binimgs = binimgs.to(device).float()
                    loss = loss_fn(preds, binimgs)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    opt.step()
                    counter += 1
                    t1 = time()
                    # kbar.update(batchi, values=[("loss", loss)])        

                    if counter % 100 == 0:
                        print(epoch, counter, loss.item())
                        writer.add_scalar('train/fold{}/loss'.format(test_index[0]), loss, counter)
                        mlflow.log_metric("train/fold{}/loss".format(test_index[0]), loss)

                    if counter % 500 == 0:
                        _, _, iou = get_batch_iou(preds, binimgs)
                        print("IoU: {}".format(iou))
                        writer.add_scalar('train/fold{}/iou'.format(test_index[0]), iou, counter)
                        # writer.add_scalar('train/epoch', epoch, counter)
                        writer.add_scalar('train/step_time', t1 - t0, counter)
                        mlflow.log_metric('train/fold{}/iou'.format(test_index[0]), iou)

                val_info = get_val_info(model, testloader, loss_fn, device)
                print('Epoch: ', epoch, 'VAL', val_info)
                test_results_iou.append(val_info['iou'])
                writer.add_scalar('val/fold{}/loss'.format(test_index[0]), val_info['loss'], counter)
                writer.add_scalar('val/fol{}/iou'.format(test_index[0]), val_info['iou'], counter)
                mlflow.log_metric('val/fold{}/loss'.format(test_index[0]), val_info['loss'])
                mlflow.log_metric('val/fold{}/iou'.format(test_index[0]), val_info['iou'])

                model.eval()
                mname = os.path.join(logdir, "model_fold{}.pt".format(test_index[0]))
                print('saving', mname)
                torch.save(model.state_dict(), mname)
                mlflow.pytorch.log_model(model, 'model_fold{}'.format(test_index[0]))
                model.train()

        print("Average iou: {}".format(sum(test_results_iou)/len(test_results_iou)))
