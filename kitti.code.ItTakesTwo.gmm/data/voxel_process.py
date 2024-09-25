import numpy
import numba as nb
import torch


class VoxelProcess:
    def __init__(self, max_volume_space, min_volume_space, intervals, voxel_grid_size, ignore_label):
        self.max_volume_space = max_volume_space
        self.min_volume_space = min_volume_space
        self.voxel_grid_size = voxel_grid_size
        self.ignore_label = ignore_label
        self.intervals = intervals
    
    # def a(self, xyz_pol, xyz):
    #     max_bound = numpy.asarray(self.max_volume_space)
    #     min_bound = numpy.asarray(self.min_volume_space)
    #     # get grid index
    #     crop_range = max_bound - min_bound
    #     cur_grid_size = self.voxel_grid_size
    #     intervals = crop_range / (cur_grid_size - 1)
    #
    #     if (intervals == 0).any(): print("Zero interval!")
    #     grid_ind = (numpy.floor((numpy.clip(xyz_pol, min_bound, max_bound) - min_bound) / intervals)).astype(numpy.int_)
    #
    #     voxel_position = numpy.zeros(self.voxel_grid_size, dtype=numpy.float32)
    #     dim_array = numpy.ones(len(self.voxel_grid_size) + 1, int)
    #     dim_array[0] = -1
    #     voxel_position = numpy.indices(self.voxel_grid_size) * intervals.reshape(dim_array) + min_bound.reshape(dim_array)
    #     # voxel_position = polar2cat(voxel_position)
    #
    #     # processed_label = numpy.ones(self.voxel_grid_size, dtype=numpy.uint8) * self.ignore_label
    #     # label_voxel_pair = numpy.concatenate([grid_ind, labels], axis=1)
    #     # label_voxel_pair = label_voxel_pair[numpy.lexsort((grid_ind[:, 0], grid_ind[:, 1], grid_ind[:, 2])), :]
    #     # processed_label = nb_process_label(numpy.copy(processed_label), label_voxel_pair)
    #     # data_tuple = (voxel_position, processed_label)
    #
    #     # center data on each voxel for PTnet
    #     voxel_centers = (grid_ind.astype(numpy.float32) + 0.5) * intervals + min_bound
    #     return_xyz = xyz_pol - voxel_centers
    #     return_xyz = numpy.concatenate((return_xyz, xyz_pol, xyz[:, :2]), axis=1)
    #     # return_fea = np.concatenate((return_xyz, sig[..., np.newaxis]), axis=1)
    #     return return_xyz

    def __call__(self, points, point_labels):
        # points = [x, y, z, intensity]
        # cart -> polar based on x,y,z
        xyz_pol = self.cart2polar(points)
        # temp = self.a(xyz_pol, points)
        # temp = numpy.concatenate((temp, points[:, -1][..., numpy.newaxis]), axis=1)
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
        processed_label = numpy.ones(self.voxel_grid_size, dtype=numpy.uint8) * self.ignore_label
        processed_label = self.nb_process_label(numpy.copy(processed_label), label_voxel_pair)

        return point_feature, point_coord, processed_label

    @staticmethod
    def cart2polar(inumpyut_xyz):
        rho = numpy.sqrt(inumpyut_xyz[:, 0] ** 2 + inumpyut_xyz[:, 1] ** 2)
        phi = numpy.arctan2(inumpyut_xyz[:, 1], inumpyut_xyz[:, 0])
        return numpy.stack((rho, phi, inumpyut_xyz[:, 2]), axis=1)

    @staticmethod
    @nb.jit('u1[:,:,:](u1[:,:,:],i8[:,:])', nopython=True, cache=True, parallel=False)
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
