import os
import numpy
from easydict import EasyDict

C = EasyDict()

C.seed = 666

""" root directory """
C.repo_name = 'lidar_segmentation'
C.root_dir = os.path.realpath("kitti.code.ItTakesTwo.gmm")

""" dataset setup """
C.data_path = "/data/semantic_kitti/"
C.num_workers = 4

C.rings = 64
C.fov_up = 3.0
C.fov_down = -25.0
C.horizontal_angular_resolution = 0.17578125
C.horizontal_resolution = int(360 / C.horizontal_angular_resolution)
C.max_volume_bound = numpy.array([50, numpy.pi, 2])
C.min_volume_bound = numpy.array([0, -numpy.pi, -4])

# uniform sampling setup, following LaserMix: https://arxiv.org/abs/2207.00026
C.feat_compression = 16
C.grid_size = numpy.array([240, 180, 20])

# note:
# please uncomment the following lines if you use partial, significant sampling protocols.
# C.feat_compression = 32
# C.grid_size = numpy.array([480, 360, 32])
C.intervals = (C.max_volume_bound - C.min_volume_bound) / (C.grid_size - 1)

""" model setup """
C.n_classes = 20
C.init_size = 32
C.ignore_index = 0
C.feat_dimension = 9
C.output_feature = 256

""" contrastive loss """
C.embeds_num = 100
C.embeds_dim = 64
C.temperature = .1
C.epsilon = 1e-16

""" optimiser """
C.betas = [0.9, 0.999]
C.weight_decay = 0.001

""" training setup """
C.lr = 1e-3
C.batch_size = 8
C.lr_power = 0.9
C.epochs = 60


""" wandb setup """
# Specify you wandb environment KEY; and paste here
C.wandb_key = ""

# Your project [work_space] name
C.project_name = "IT2"

C.experiment_name = "kitti.final.10pct"

# False for debug; True for visualize
C.wandb_online = True

""" save setup """
C.path_to_saved_dir = "./ckpts/exp"
C.ckpts_dir = os.path.join(C.path_to_saved_dir, C.experiment_name)

import pathlib
pathlib.Path(C.ckpts_dir).mkdir(parents=True, exist_ok=True)

