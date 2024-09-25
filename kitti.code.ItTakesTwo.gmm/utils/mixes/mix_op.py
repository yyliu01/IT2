import torch
from utils.mixes.range_mix import RangeMix
from utils.mixes.voxel_mix import VoxelMix


class CutMix(torch.nn.Module):
    def __init__(self,  param):
        super().__init__()
        self.param = param
        self.range_view_mix = RangeMix(range_mix_factor=4)
        self.voxel_view_mix = VoxelMix(batch_size=param.batch_size, radius_keep=20,
                                       grid_size=param.grid_size, ignore_index=param.ignore_index)
        # self.img_size = crop_size
        # self.bbox_fetfch = self.rand_bbox_1 if n_boxes == 1 else NotImplementedError

    def forward(self, *mix_info, modality):
        assert modality in ["range", "voxel"], "modality error."

        return self.range_view_mix(*mix_info) if modality == "range" else \
            self.voxel_view_mix(*mix_info)
