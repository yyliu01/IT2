import torch


class VoxelMix(torch.nn.Module):
    def __init__(self, batch_size, grid_size, radius_keep, ignore_index):
        super().__init__()
        self.batch_size = batch_size
        self.grid_size = grid_size
        self.ignore_index = ignore_index
        self.radius_keep = radius_keep

    def forward(self, point_feature_, point_coord_, voxel_label_):
        pc_lookup = point_coord_.clone()
        vl_lookup = voxel_label_.clone()

        cut_num = 8
        u_rand_index = torch.stack([torch.randperm(self.batch_size).cuda() for i in range(cut_num)])
        cake_slice = 360 // cut_num

        for area_idx in range(u_rand_index.shape[0]):
            st_degree = area_idx * cake_slice
            ed_degree = st_degree + cake_slice
            st_degree = st_degree // 2
            ed_degree = ed_degree // 2
            for batch_idx, perm_idx in enumerate(u_rand_index[area_idx]):
                rep_cond = torch.where(
                    (pc_lookup[:, 0] == perm_idx) &
                    (
                            (pc_lookup[:, 2] >= st_degree) &
                            (pc_lookup[:, 2] < ed_degree) &
                            (pc_lookup[:, 1] >= self.radius_keep)
                    )
                )
                point_coord_[rep_cond[0], 0] = batch_idx

                # Assign perm_idx's voxel grid label to current sample
                voxel_label_[batch_idx, self.radius_keep:, st_degree:ed_degree, :] \
                    = vl_lookup[perm_idx, self.radius_keep:, st_degree:ed_degree, :]

        del pc_lookup, vl_lookup
        return point_feature_, point_coord_, voxel_label_
