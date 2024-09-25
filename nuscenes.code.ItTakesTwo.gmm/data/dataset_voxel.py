import os
from typing import Tuple

import numba
import numpy
import torch
import torch.utils.data as data
from joblib import Memory

from data.lidar import LidarSegDatabaseInterface
from data.utils import get_range_view_inputs


class LidarSegVoxelDataset(data.Dataset):

    def __init__(
            self,
            db: LidarSegDatabaseInterface,
            max_volume_space: numpy.array,
            min_volume_space: numpy.array,
            intervals: numpy.array,
            voxel_grid_size: numpy.array,
            n_classes: int = 17,
            ignore_index: int = 0,
            augment: bool = False,
            horiz_angular_res: float = 0.1875,
            cache_dir: str = "./cache/"
    ):
        self.db = db
        self.intervals = intervals
        self.tokens = self.db.tokens
        self.voxel_grid_size = voxel_grid_size
        self.max_volume_space = max_volume_space  # [50, numpy.pi, 3]
        self.min_volume_space = min_volume_space  # [0, -numpy.pi, -5]
        self.horiz_angular_res = horiz_angular_res

        self.n_classes = n_classes
        self.ignore_index = ignore_index
        self.cache_dir = cache_dir

        # if self.cache_dir is not None:
        #     self.memory = Memory(self.cache_dir, verbose=False, compress=False)
        #     # ignore=['self'] flag is so that not the whole object (along with the input parameters) is hashed
        #     self.load_data = self.memory.cache(self.load_data, ignore=['self'])

        self.augment = augment

    def __len__(self) -> int:
        return len(self.db.tokens)

    def __getitem__(self, index: int) -> Tuple:
        token = self.db.tokens[index]
        point_, point_label_, rings = self.load_data(token)
        # (25694, 4) (25694,) (25694,)
        # apply augmentation
        point_ = self.augmentation(point_) if self.augment else point_
        point_feature_, point_coord_, voxel_label_ = \
            self.voxel_process(point_, point_label_)

        # return rv, label_rv, idx_rv, numpy.stack((proj_x, proj_y), axis=1)
        range_feature, range_label, range_idx, proj_coord = \
            self.range_process(point_, point_label_, rings)

        return point_feature_, point_coord_, point_label_, voxel_label_, range_feature, range_label, proj_coord, token

    def load_data(self, token: str) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
        return self.db.load_from_db(token)

    def range_process(self, points, point_labels, rings):
        proj_y, proj_x, ht, wt = self.db.load_range_view_coordinates(points, ring=rings,
                                                                     horiz_angular_res=self.horiz_angular_res)
        rv, label_rv, idx_rv = get_range_view_inputs(points=points,
                                                     points_label=point_labels,
                                                     proj_y=proj_y, proj_x=proj_x, ht=ht, wt=wt,
                                                     add_binary_mask=True,
                                                     ignore_label=self.db.local2id['ignore_label'])

        rv = torch.from_numpy(rv).permute(2, 0, 1).float()
        # we don't have third dimension of range view
        return rv, label_rv, idx_rv, numpy.stack((proj_x, proj_y), axis=1)

    def voxel_process(self, points, point_labels):
        # points = [x, y, z, intensity]
        # cart -> polar based on x,y,z
        xyz_pol = self.cart2polar(points)
        point_labels = numpy.expand_dims(point_labels, axis=1)

        point_coord = (numpy.floor((numpy.clip(xyz_pol, self.min_volume_space,
                                               self.max_volume_space)
                                    - self.min_volume_space) / self.intervals)).astype(numpy.int_)

        # dim_array = numpy.ones(len(self.voxel_grid_size) + 1, int)
        # dim_array[0] = -1
        # voxel_position = numpy.indices(self.voxel_grid_size) * self.intervals.reshape(dim_array) +\
        #     self.min_volume_space.reshape(dim_array)

        voxel_center = (point_coord.astype(numpy.float32) + 0.5) * self.intervals + self.min_volume_space
        point_voxel_centers = xyz_pol - voxel_center
        point_feature = numpy.concatenate([point_voxel_centers, xyz_pol, points[:, :2], points[:, 3:]], axis=1)

        label_voxel_pair = numpy.concatenate([point_coord, point_labels], axis=1)
        label_voxel_pair = label_voxel_pair[numpy.lexsort((point_coord[:, 0], point_coord[:, 1], point_coord[:, 2])), :]
        processed_label = numpy.ones(self.voxel_grid_size, dtype=numpy.uint8) * self.ignore_index
        processed_label = self.nb_process_label(numpy.copy(processed_label), label_voxel_pair)

        return point_feature, point_coord, processed_label

    @staticmethod
    @numba.jit('u1[:,:,:](u1[:,:,:],i8[:,:])', nopython=True, cache=True, parallel=False)
    def nb_process_label(processed_label, sorted_label_voxel_pair):
        label_size = 256
        counter = numpy.zeros((label_size,), dtype=numpy.uint16)
        counter[sorted_label_voxel_pair[0, 3]] = 1
        cur_sear_ind = sorted_label_voxel_pair[0, :3]
        for i in range(1, sorted_label_voxel_pair.shape[0]):
            cur_ind = sorted_label_voxel_pair[i, :3]
            if not numpy.all(numpy.equal(cur_ind, cur_sear_ind)):
                processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = numpy.argmax(counter)
                counter = numpy.zeros((label_size,), dtype=numpy.uint16)
                cur_sear_ind = cur_ind
            counter[sorted_label_voxel_pair[i, 3]] += 1
        processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = numpy.argmax(counter)
        return processed_label

    @staticmethod
    def cart2polar(input_xyz):
        rho = numpy.sqrt(input_xyz[:, 0] ** 2 + input_xyz[:, 1] ** 2)
        phi = numpy.arctan2(input_xyz[:, 1], input_xyz[:, 0])
        return numpy.stack((rho, phi, input_xyz[:, 2]), axis=1)

    @staticmethod
    def polar2cart(input_xyz_polar):
        x = input_xyz_polar[0] * numpy.cos(input_xyz_polar[1])
        y = input_xyz_polar[0] * numpy.sin(input_xyz_polar[1])
        return numpy.stack((x, y, input_xyz_polar[2]), axis=0)

    @staticmethod
    def polar2cart_point(input_xyz_polar):
        x = input_xyz_polar[:, 0] * numpy.cos(input_xyz_polar[:, 1])
        y = input_xyz_polar[:, 0] * numpy.sin(input_xyz_polar[:, 1])
        return numpy.stack((x, y, input_xyz_polar[:, 2]), axis=1)

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
        token = [d[-1] for d in data_]

        for i in range(batch_size):
            # the batch_padding format is different between spconv and torchsparse
            point_coord[i] = numpy.pad(point_coord[i], ((0, 0), (1, 0)),
                                       mode='constant', constant_values=i)
            project_coord[i] = numpy.pad(project_coord[i], ((0, 0), (1, 0)),
                                         mode='constant', constant_values=i)

        batch_data['point_feature'] = torch.from_numpy(numpy.concatenate(point_feature)).type(torch.FloatTensor)
        batch_data['point_coord'] = torch.from_numpy(numpy.concatenate(point_coord)).type(torch.LongTensor)
        batch_data['point_label'] = torch.from_numpy(numpy.concatenate(point_label)).type(torch.LongTensor)
        batch_data['voxel_label'] = torch.from_numpy(numpy.stack(voxel_label)).type(torch.LongTensor)
        batch_data['range_feature'] = torch.from_numpy(numpy.stack(range_feature))
        batch_data['range_label'] = torch.from_numpy(numpy.stack(range_label)).type(torch.LongTensor)
        batch_data['project_coord'] = torch.from_numpy(numpy.concatenate(project_coord)).type(torch.LongTensor)
        batch_data['offset'] = torch.cumsum(torch.tensor([sample[0].shape[0] for sample in data_],
                                                         dtype=torch.long), dim=0)
        # batch_data['token'] = [token]
        # batch_data['unique_point_coord'] = torch.unique(batch_data['point_coord'], dim=0)
        return batch_data

    @staticmethod
    def augmentation(points):
        rotate_rad = numpy.deg2rad(numpy.random.random() * 360) - numpy.pi
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

        # convert coordinate into polar coordinates
        trans_std = [0.1, 0.1, 0.1]
        noise_translate = numpy.array([numpy.random.normal(0, trans_std[0], 1),
                                       numpy.random.normal(0, trans_std[1], 1),
                                       numpy.random.normal(0, trans_std[2], 1)]).T
        points[:, 0:3] += noise_translate

        return points
