import numpy as np
import torch


class Metrics:
    def __init__(self, n_classes: int, ignore_index: int = 0, device: str = "cpu"):
        self.labelled = IouEval(n_classes, ignore_index, device_=device)
        self.unlabelled = IouEval(n_classes, ignore_index, device_=device)
        self.valid = IouEval(n_classes, ignore_index, device_=device)

    def reset(self):
        self.valid.reset()
        self.labelled.reset()
        self.unlabelled.reset()


class IouEval(torch.nn.Module):
    def __init__(self, n_classes: int, ignore_index: int = 0, device_: str = "cpu"):
        super(IouEval, self).__init__()
        self.n_classes = n_classes
        self.device = device_

        self.ignore = torch.tensor(ignore_index).long()
        self.include = torch.tensor([n for n in range(self.n_classes) if n not in self.ignore]).long().to(self.device)
        self.conf_matrix = torch.zeros((self.n_classes, self.n_classes), dtype=torch.long).to(self.device)
        self.ones = None
        self.last_scan_size = None

    def num_classes(self):
        return self.n_classes

    def add_batch(self, x, batch_info, branch_name):
        if branch_name == "voxel":
            x = x[batch_info['point_coord'][:, 0], batch_info['point_coord'][:, 1], batch_info['point_coord'][:, 2],
                  batch_info['point_coord'][:, 3]]
            y = batch_info['point_label']
        elif branch_name == "range":
            y = batch_info['range_label']
        else:
            x = x
            y = batch_info['point_label']

        x = x.to(self.device)
        y = y.to(self.device)

        x_row = x.reshape(-1)
        y_row = y.reshape(-1)

        idxs = torch.stack([x_row, y_row], dim=0)

        if self.ones is None or self.last_scan_size != idxs.shape[-1]:
            self.ones = torch.ones((idxs.shape[-1]), device=x.device).long()
            self.last_scan_size = idxs.shape[-1]
        self.conf_matrix = self.conf_matrix.index_put_(tuple(idxs), self.ones, accumulate=True)

    def get_stats(self):
        conf = self.conf_matrix.clone().double()
        conf[self.ignore] = 0
        conf[:, self.ignore] = 0

        tp = conf.diag()
        fp = conf.sum(dim=1) - tp
        fn = conf.sum(dim=0) - tp
        return tp, fp, fn

    def get_iou(self):
        tp, fp, fn = self.get_stats()
        intersection = tp
        union = tp + fp + fn + 1e-15
        iou = intersection / union
        iou_mean = (intersection[self.include] / union[self.include]).mean()
        return torch.round(iou_mean, decimals=4), iou

    def get_acc(self):
        tp, fp, fn = self.get_stats()
        total_tp = tp.sum()
        total = tp[self.include].sum() + fp[self.include].sum() + 1e-15
        acc_mean = total_tp / total
        return torch.round(acc_mean, decimals=4)

    def reset(self):
        self.conf_matrix = torch.zeros((self.n_classes, self.n_classes), dtype=torch.long).to(self.device)
        self.ones = None
        self.last_scan_size = None

