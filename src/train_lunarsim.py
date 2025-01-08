import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import cv2
import os
from PIL import Image

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

def train( 
            nepochs=1,         #10000
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

            bsz=4,
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
    batch_size = 4
    num_cameras = 6

    mlflow.set_tracking_uri("http://misc.mlflow.kplabs.pl")
    mlflow.set_experiment("BEVER LSS on LunarSim")

    data_traverse0 = "./lunarsim_bev/traverse0/"
    target_traverse0 = "./lunarsim_bev/traverse0/bev_seg/"
    traverse0_paths, traverse0_target_paths = get_traverse_paths(data_traverse0, target_traverse0) 

    data_traverse1 = "./lunarsim_bev/traverse1/"
    target_traverse1 = "./lunarsim_bev/traverse1/bev_seg/"
    traverse1_paths, traverse1_target_paths = get_traverse_paths(data_traverse1, target_traverse1) 

    data_traverse2 = "./lunarsim_bev/traverse2/"
    target_traverse2 = "./lunarsim_bev/traverse2/bev_seg/"
    traverse2_paths, traverse2_target_paths = get_traverse_paths(data_traverse2, target_traverse2)

    data_traverse3 = "./lunarsim_bev/traverse3/"
    target_traverse3 = "./lunarsim_bev/traverse3/bev_seg/"
    traverse3_paths, traverse3_target_paths = get_traverse_paths(data_traverse3, target_traverse3) 

    data_traverse4 = "./lunarsim_bev/traverse4/"
    target_traverse4 = "./lunarsim_bev/traverse4/bev_seg/"
    traverse4_paths, traverse4_target_paths = get_traverse_paths(data_traverse4, target_traverse4) 

    data_traverse5 = "./lunarsim_bev/traverse5/"
    target_traverse5 = "./lunarsim_bev/traverse5/bev_seg/"
    traverse5_paths, traverse5_target_paths = get_traverse_paths(data_traverse5, target_traverse5) 

    data_traverse6 = "./lunarsim_bev/traverse6/"
    target_traverse6 = "./lunarsim_bev/traverse6/bev_seg/"
    traverse6_paths, traverse6_target_paths = get_traverse_paths(data_traverse6, target_traverse6) 

    traverse2_paths.extend(traverse3_paths)
    traverse2_target_paths.extend(traverse3_target_paths)
    traverse2_paths.extend(traverse4_paths)
    traverse2_target_paths.extend(traverse4_target_paths)
    traverse2_paths.extend(traverse5_paths)
    traverse2_target_paths.extend(traverse5_target_paths)
    traverse2_paths.extend(traverse6_paths)
    traverse2_target_paths.extend(traverse6_target_paths)

    X_test = traverse0_paths
    y_test = traverse0_target_paths

    X_val = traverse1_paths
    y_val = traverse1_target_paths

    X_train = traverse2_paths
    y_train = traverse2_target_paths

    print(len(X_train))
    print(len(X_test))
    print(len(X_val))

    # Define the data and data loaders
    train_data = LunarsimDataAug(y_train, X_train, img_size, batch_size, num_cameras, data_aug_conf)
    val_data = LunarsimData(y_val, X_val, img_size, batch_size, num_cameras)
    test_data = LunarsimData(y_test, X_test, img_size, batch_size, num_cameras)
    
    trainloader = DataLoader(train_data, batch_size=4, shuffle=True, num_workers=10)
    valloader = DataLoader(val_data, batch_size=4, shuffle=True, num_workers=10)

    gpuid=0
    device = torch.device('cpu') if gpuid < 0 else torch.device(f'cuda:{gpuid}')
    model = compile_model(grid_conf, data_aug_conf, outC=1)
    model.to(device)
    mlflow.pytorch.autolog()

            # run the experiment
    with mlflow.start_run(run_name="Bever test run with mobile cam encode"):

        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        # testing smaller learning rate
        # opt = torch.optim.Adam(model.parameters(), lr=lr)

        loss_fn = SimpleLoss(pos_weight).cuda(gpuid)
        # loss_fn = DiceLoss().cuda(gpuid)

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


                if counter % 10 == 0:
                    print(epoch, counter, loss.item())
                    writer.add_scalar('train/loss', loss, counter)
                    mlflow.log_metric("train/loss", loss)

                if counter % 50 == 0:
                    _, _, iou = get_batch_iou(preds, binimgs)
                    print("IoU: {}".format(iou))
                    writer.add_scalar('train/iou', iou, counter)
                    writer.add_scalar('train/epoch', epoch, counter)
                    writer.add_scalar('train/step_time', t1 - t0, counter)
                    mlflow.log_metric('train/iou', iou)
                    # kbar.update(batchi, values=[("iou", iou)]) 

                if counter % 200 == 0:
                    viz_preds(preds, binimgs)



            val_info = get_val_info(model, valloader, loss_fn, device)
            print('Epoch: ', epoch, 'VAL', val_info)
            writer.add_scalar('val/loss', val_info['loss'], counter)
            writer.add_scalar('val/iou', val_info['iou'], counter)
            mlflow.log_metric('val/loss', val_info['loss'])
            mlflow.log_metric('val/iou', val_info['iou'])
            # kbar.add(1, values=[("val_loss", val_info['loss']), ("val_iou", val_info['iou'])])
            # save the model if iou is grater than previously
            if val_info['iou'] > iou_loss_checkpoint:
                iou_loss_checkpoint = val_info['iou']
                model.eval()
                mname = os.path.join(logdir, "model.pt".format(counter))
                print('saving', mname)
                torch.save(model.state_dict(), mname)
                mlflow.pytorch.log_model(model, 'model')
                model.train()
