"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""

from fire import Fire

import src
from src import train_lunarsim
from src import train_lunarsim_kfold


if __name__ == '__main__':
    Fire({
        'lidar_check': src.explore.lidar_check,
        'cumsum_check': src.explore.cumsum_check,

        'train_lunarsim': src.train_lunarsim.train,
        'train_lunarsim_kfold': src.train_lunarsim_kfold.train_cv,
        'eval_model_iou': src.explore.run_model,
        'viz_model_preds': src.explore.viz_model_preds,
    })
