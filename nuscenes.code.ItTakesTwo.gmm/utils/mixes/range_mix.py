import numpy
import torch


class RangeMix(torch.nn.Module):
    def __init__(self, range_mix_factor):
        super().__init__()
        self.range_mix_factor = range_mix_factor

    def rand_bbox(self, size, lam=None):
        b, _, h, w = size

        bbx_w_arrays = torch.cumsum(torch.randint(size=[b, self.range_mix_factor + 1], low=0,
                                                  high=int(w / (self.range_mix_factor + 1))), dim=1)
        bbw_1, bbw_2 = bbx_w_arrays[:, :self.range_mix_factor], bbx_w_arrays[:, 1:]

        cut_h = int(h * numpy.sqrt(1. - lam))
        bbx_h_arrays = torch.randint(size=[b, self.range_mix_factor + 1], low=int(h / 8), high=h)
        bbh_1, bbh_2 = torch.clip(bbx_h_arrays - cut_h // 2, 0, h), torch.clip(bbx_h_arrays + cut_h // 2, 0, h)

        return bbw_1, bbw_2, bbh_1, bbh_2

    def forward(self, range_scan, project_coord, range_embed, pred_label):  # real_label
        mix_project_coord, mix_range_embed = [], []
        mix_range_scan = range_scan.clone()
        mix_pred_label = pred_label.clone()

        u_rand_index = torch.randperm(mix_range_scan.size()[0])
        bbw_1_, bbw_2_, bbh_1_, bbh_2_ = self.rand_bbox(range_scan.size(), lam=numpy.random.beta(4, 4))
        for idx in range(0, range_scan.shape[0]):
            instance_cut, instance_mix = [], []
            cond_mix = (project_coord[:, 0] == idx)
            cond_org = (project_coord[:, 0] == u_rand_index[idx])

            for bbox in range(0, self.range_mix_factor):
                mix_range_scan[idx, :, bbh_1_[idx][bbox]:bbh_2_[idx][bbox], bbw_1_[idx][bbox]:bbw_2_[idx][bbox]] = \
                    range_scan[u_rand_index[idx], :, bbh_1_[idx][bbox]:bbh_2_[idx][bbox],
                    bbw_1_[idx][bbox]:bbw_2_[idx][bbox]]

                mix_pred_label[idx, bbh_1_[idx][bbox]:bbh_2_[idx][bbox], bbw_1_[idx][bbox]:bbw_2_[idx][bbox]] = \
                    pred_label[u_rand_index[idx], bbh_1_[idx][bbox]:bbh_2_[idx][bbox],
                    bbw_1_[idx][bbox]:bbw_2_[idx][bbox]]

                cond_h = torch.logical_and((project_coord[:, 2] >= bbh_1_[idx][bbox]),
                                           (project_coord[:, 2] < bbh_2_[idx][bbox]))
                cond_w = torch.logical_and((project_coord[:, 1] >= bbw_1_[idx][bbox]),
                                           (project_coord[:, 1] < bbw_2_[idx][bbox]))
                cond_box = torch.logical_and(cond_h, cond_w)

                """ Mixup Project Coord """
                mix_idx = torch.logical_and(cond_mix, cond_box)
                instance_mix.append(mix_idx)

                """ Target Project Coord """
                org_idx = torch.logical_and(cond_org, cond_box)
                instance_cut.append(org_idx)

            instance_mix_idx = torch.any(torch.stack(instance_mix), dim=0)
            instance_cut_idx = torch.any(torch.stack(instance_cut), dim=0)

            instance_mix_idx = torch.logical_xor(cond_mix, instance_mix_idx)

            instance_mix = project_coord[instance_mix_idx, :]
            instance_cut = project_coord[instance_cut_idx, :]
            instance_cut[:, 0] = idx

            mix_project_coord.append(torch.cat((instance_mix, instance_cut), dim=0))

            """ Mixup Range Embedding"""
            instance_mix_emb = range_embed[instance_mix_idx, :]
            instance_cut_emb = range_embed[instance_cut_idx, :]
            mix_range_embed.append(torch.cat((instance_mix_emb, instance_cut_emb), dim=0))

        del range_scan, pred_label
        return mix_range_scan, torch.cat(mix_project_coord), mix_pred_label, torch.cat(mix_range_embed)