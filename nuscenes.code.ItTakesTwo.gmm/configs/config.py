import os
import numpy
from easydict import EasyDict

C = EasyDict()

C.seed = 666

""" root directory """
C.repo_name = 'lidar_segmentation'
C.root_dir = os.path.realpath("nuscenes.code.ItTakesTwo.gmm")

""" dataset setup """
C.data_path_mini = "/path/to/mini-demo"

C.data_path_full = "/path/to/full/dataset"

C.num_workers = 2

C.rings = 32
C.resolution = 0.1
C.min_distance = 0.9
C.horizontal_angular_resolution = 0.1875
C.horizontal_resolution = int(360 / C.horizontal_angular_resolution)
C.grid_size = numpy.array([240, 180, 20])
C.max_volume_bound = numpy.array([50, numpy.pi, 3])
C.min_volume_bound = numpy.array([0, -numpy.pi, -5])
C.intervals = (C.max_volume_bound - C.min_volume_bound) / (C.grid_size - 1)

""" contrastive loss """
C.embeds_num = 100
C.embeds_dim = 64
C.temperature = .1
C.epsilon = 1e-16

""" model setup """
C.n_classes = 17
C.init_size = 16
C.ignore_index = 0
C.feat_dimension = 9
C.feat_compression = 16
C.output_feature = 256

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

C.experiment_name = "nuscenec.final.10pct"

# False for debug; True for visualize
C.wandb_online = True

""" save setup """
C.path_to_saved_dir = "./ckpts/exp"
C.path_to_cache_dir = ".cache/"
C.ckpts_dir = os.path.join(C.path_to_saved_dir, C.experiment_name)
C.cache_dir = os.path.join(C.path_to_saved_dir, C.experiment_name)

import pathlib
pathlib.Path(C.ckpts_dir).mkdir(parents=True, exist_ok=True)
pathlib.Path(C.cache_dir).mkdir(parents=True, exist_ok=True)

