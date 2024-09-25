import os

import numpy
import torch
from torch.utils import data

from data.range_process import RangeProcess
from data.voxel_process import VoxelProcess


class SemanticKittiDatabase(data.Dataset):
    def __init__(self, root: str, partition: int, split: str, mode: str, data_dir: str,
                 range_info: tuple, voxel_info: tuple, augment: bool = False):

        if split == "train":
            file_list = os.path.join(root, 'data', 'splits', f"{partition}pct", f"{mode}_list.txt")
        elif split == "valid":
            file_list = os.path.join(root, 'data', 'splits', f"{split}_list.txt")
        else:
            raise FileNotFoundError

        self.input_file = [os.path.join(data_dir, "kitti_input", split, line.rstrip().split('/')[0], "velodyne",
                                        line.rstrip().split('/')[1] + ".bin") for line in tuple(open(file_list, "r"))]
        # scribble_label, kitti_label
        # scribble setup
        label_name = "kitti_label" 
        # label_name = "kitti_label" if  split == "valid" else "scribble_label"
        self.label_file = [os.path.join(data_dir, label_name, split, line.rstrip().split('/')[0], "labels",
                                        line.rstrip().split('/')[1] + ".label") for line in tuple(open(file_list, "r"))]
        self.learning_map = labels_map
        self.augment = augment

        self.voxel_process = VoxelProcess(*voxel_info)
        self.range_process = RangeProcess(*range_info)

    def __len__(self):
        return len(self.label_file)

    def load_data(self, index_: int):
        points_ = numpy.fromfile(self.input_file[index_], dtype=numpy.float32).reshape((-1, 4))
        labels_ = numpy.fromfile(self.label_file[index_], dtype=numpy.uint32).reshape((-1, 1))
        labels_ = labels_ & 0xFFFF  # delete high 16 digits binary
        labels_ = numpy.vectorize(self.learning_map.__getitem__)(labels_)
        return points_, labels_.squeeze()

    def __getitem__(self, index: int):
        points, labels = self.load_data(index)
        points = self.augmentation(points) if self.augment else points
        point_feature, point_coord, voxel_label = self.voxel_process(points, labels)
        range_feature, range_label, proj_coord, cover_map = self.range_process(points, labels)

        return point_feature, point_coord, labels, voxel_label, range_feature, range_label, proj_coord, cover_map

    @staticmethod
    def collate_batch(data_):
        batch_size = len(data_)
        batch_data = {}
        point_feature = [d[0] for d in data_]
        point_coord = [d[1] for d in data_]
        point_label = [d[2] for d in data_]
        voxel_label = [d[3] for d in data_]
        range_feature = [d[4] for d in data_]
        range_label = [d[5] for d in data_]
        project_coord = [d[6] for d in data_]
        # idx_rv = [d[7] for d in data_]
        cover_map = [d[-1] for d in data_]

        for i in range(batch_size):
            # the batch_padding format is different between spconv and torchsparse
            point_coord[i] = numpy.pad(point_coord[i], ((0, 0), (1, 0)),
                                       mode='constant', constant_values=i)
            project_coord[i] = numpy.pad(project_coord[i], ((0, 0), (1, 0)),
                                         mode='constant', constant_values=i)
            # idx_rv[i] = numpy.pad(idx_rv[i], ((0, 0), (1, 0)),
            #                              mode='constant', constant_values=i)

        batch_data['point_feature'] = torch.from_numpy(numpy.concatenate(point_feature)).type(torch.FloatTensor)
        batch_data['point_coord'] = torch.from_numpy(numpy.concatenate(point_coord)).type(torch.LongTensor)
        batch_data['point_label'] = torch.from_numpy(numpy.concatenate(point_label)).type(torch.LongTensor)
        batch_data['voxel_label'] = torch.from_numpy(numpy.stack(voxel_label)).type(torch.LongTensor)
        batch_data['range_feature'] = torch.from_numpy(numpy.stack(range_feature))
        batch_data['range_label'] = torch.from_numpy(numpy.stack(range_label)).type(torch.LongTensor)
        batch_data['project_coord'] = torch.from_numpy(numpy.concatenate(project_coord)).type(torch.LongTensor)
        # batch_data['range_index'] = torch.from_numpy(numpy.stack(idx_rv)).type(torch.LongTensor)
        batch_data['cover_map'] = torch.from_numpy(numpy.concatenate(cover_map)).type(torch.bool)
        batch_data['offset'] = torch.cumsum(torch.tensor([sample[0].shape[0] for sample in data_],
                                                         dtype=torch.long), dim=0)
        return batch_data

    @staticmethod
    # https://github.com/xinge008/Cylinder3D/blob/30a0abb2ca4c657a821a5e9a343934b0789b2365/dataloader/dataset_semantickitti.py#L201
    def augmentation(points):
        # rotate augmentation
        rotate_rad = numpy.deg2rad(numpy.random.random() * 90) - numpy.pi/4
        c, s = numpy.cos(rotate_rad), numpy.sin(rotate_rad)
        j = numpy.matrix([[c, s], [-s, c]])
        points[:, :2] = numpy.dot(points[:, :2], j)

        # random data augmentation by flip x , y or x+y
        flip_type = numpy.random.choice(4, 1)
        if flip_type == 1:
            points[:, 0] = -points[:, 0]
        elif flip_type == 2:
            points[:, 1] = -points[:, 1]
        elif flip_type == 3:
            points[:, :2] = -points[:, :2]

        noise_scale = numpy.random.uniform(0.95, 1.05)
        points[:, 0] = noise_scale * points[:, 0]
        points[:, 1] = noise_scale * points[:, 1]

        noise_translate = numpy.array([numpy.random.normal(0, 0.1, 1),
                                       numpy.random.normal(0, 0.1, 1),
                                       numpy.random.normal(0, 0.1, 1)]).T
        points[:, 0:3] += noise_translate

        return points


"""
categories information
"""
labels_map = {
    0: 0,     # "unlabeled"
    1: 0,     # "outlier" mapped to "unlabeled" --------------------------mapped
    10: 1,     # "car"
    11: 2,     # "bicycle"
    13: 5,     # "bus" mapped to "other-vehicle" --------------------------mapped
    15: 3,     # "motorcycle"
    16: 5,     # "on-rails" mapped to "other-vehicle" ---------------------mapped
    18: 4,     # "truck"
    20: 5,     # "other-vehicle"
    30: 6,     # "person"
    31: 7,     # "bicyclist"
    32: 8,     # "motorcyclist"
    40: 9,     # "road"
    44: 10,    # "parking"
    48: 11,    # "sidewalk"
    49: 12,    # "other-ground"
    50: 13,    # "building"
    51: 14,    # "fence"
    52: 0,     # "other-structure" mapped to "unlabeled" ------------------mapped
    60: 9,     # "lane-marking" to "road" ---------------------------------mapped
    70: 15,    # "vegetation"
    71: 16,    # "trunk"
    72: 17,    # "terrain"
    80: 18,    # "pole"
    81: 19,    # "traffic-sign"
    99: 0,     # "other-object" to "unlabeled" ----------------------------mapped
    252: 1,    # "moving-car" to "car" ------------------------------------mapped
    253: 7,    # "moving-bicyclist" to "bicyclist" ------------------------mapped
    254: 6,    # "moving-person" to "person" ------------------------------mapped
    255: 8,    # "moving-motorcyclist" to "motorcyclist" ------------------mapped
    256: 5,    # "moving-on-rails" mapped to "other-vehicle" --------------mapped
    257: 5,    # "moving-bus" mapped to "other-vehicle" -------------------mapped
    258: 4,    # "moving-truck" to "truck" --------------------------------mapped
    259: 5,    # "moving-other"-vehicle to "other-vehicle" ----------------mapped
}
"""

labels_map = {
    0: 19,  # "unlabeled"
    1: 19,  # "outlier" mapped to "unlabeled" --------------mapped
    10: 0,  # "car"
    11: 1,  # "bicycle"
    13: 4,  # "bus" mapped to "other-vehicle" --------------mapped
    15: 2,  # "motorcycle"
    16: 4,  # "on-rails" mapped to "other-vehicle" ---------mapped
    18: 3,  # "truck"
    20: 4,  # "other-vehicle"
    30: 5,  # "person"
    31: 6,  # "bicyclist"
    32: 7,  # "motorcyclist"
    40: 8,  # "road"
    44: 9,  # "parking"
    48: 10,  # "sidewalk"
    49: 11,  # "other-ground"
    50: 12,  # "building"
    51: 13,  # "fence"
    52: 19,  # "other-structure" mapped to "unlabeled" ------mapped
    60: 8,  # "lane-marking" to "road" ---------------------mapped
    70: 14,  # "vegetation"
    71: 15,  # "trunk"
    72: 16,  # "terrain"
    80: 17,  # "pole"
    81: 18,  # "traffic-sign"
    99: 19,  # "other-object" to "unlabeled" ----------------mapped
    252: 0,  # "moving-car" to "car" ------------------------mapped
    253: 6,  # "moving-bicyclist" to "bicyclist" ------------mapped
    254: 5,  # "moving-person" to "person" ------------------mapped
    255: 7,  # "moving-motorcyclist" to "motorcyclist" ------mapped
    256: 4,  # "moving-on-rails" mapped to "other-vehic------mapped
    257: 4,  # "moving-bus" mapped to "other-vehicle" -------mapped
    258: 3,  # "moving-truck" to "truck" --------------------mapped
    259: 4  # "moving-other"-vehicle to "other-vehicle"-----mapped
}

"""
class_names = [
    'car', 'bicycle', 'motorcycle', 'truck', 'bus', 'person', 'bicyclist',
    'motorcyclist', 'road', 'parking', 'sidewalk', 'other-ground', 'building',
    'fence', 'vegetation', 'trunck', 'terrian', 'pole', 'traffic-sign', 'ignore'
]
