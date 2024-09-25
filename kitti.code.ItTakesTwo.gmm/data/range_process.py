import numpy
import torch


class RangeProcess:
    def __init__(self, proj_fov_up, proj_fov_down, num_rings, horiz_angular_res, ignore_label):
        self.proj_fov_up = proj_fov_up
        self.proj_fov_down = proj_fov_down
        self.num_rings = num_rings
        self.horiz_angular_res = horiz_angular_res
        self.ignore_label = ignore_label

    @staticmethod
    def get_range_view_inputs(
            points: numpy.ndarray,
            points_label: numpy.ndarray,
            proj_y: numpy.ndarray,
            proj_x: numpy.ndarray,
            ht: int,
            wt: int,
            add_binary_mask: bool = True,
            ignore_label: int = -100,
            fill_value: int = -1
    ):
        # order the points in decreasing depth
        depth = numpy.linalg.norm(points[:, :2], axis=1)  # [n,]
        indices = numpy.arange(points.shape[0])  # [n,]
        order = numpy.argsort(depth)[::-1]  # [n,]

        depth = depth[order]
        # ori_indices = indices.copy()
        indices = indices[order]
        points = points[order]
        old_points_label = points_label.copy()
        old_proj_x = proj_x.copy()
        old_proj_y = proj_y.copy()
        proj_y = proj_y[order]
        proj_x = proj_x[order]

        feat_dim = points.shape[1]
        range_view = numpy.full((ht, wt, feat_dim + 1), fill_value, dtype=numpy.float32)  # [32, 1920, 5]
        idx_rv = numpy.full((ht, wt), fill_value, dtype=numpy.int32)  # [32, 1920]
        idx_rv[proj_y, proj_x] = indices

        range_view[proj_y, proj_x, :feat_dim] = points  # x-y-z-intensity-(otherfeatures) as channels
        range_view[proj_y, proj_x, feat_dim] = depth  # add depth channel

        if add_binary_mask:
            range_view = numpy.concatenate((range_view, numpy.atleast_3d((idx_rv != fill_value).astype(numpy.int32))),
                                           axis=2)

        if points_label is not None:
            points_label = points_label[order]
            label_rv = numpy.full((ht, wt), ignore_label, dtype=numpy.int64)
            label_rv[proj_y, proj_x] = points_label

        else:
            label_rv = None
        temp = label_rv[old_proj_y, old_proj_x]
        cover_map = numpy.array((temp == old_points_label), dtype=bool)
        return range_view, label_rv, idx_rv, cover_map

    def __call__(self, points, labels):
        proj_y, proj_x, ht, wt = self.do_range_project(points, int(360 / self.horiz_angular_res), self.num_rings)

        rv, label_rv, idx_rv, cover_map = self.get_range_view_inputs(points=points,
                                                                        points_label=labels,
                                                                        proj_y=proj_y, proj_x=proj_x, ht=ht, wt=wt,
                                                                        add_binary_mask=True,
                                                                        ignore_label=self.ignore_label)

        rv = torch.from_numpy(rv).permute(2, 0, 1).float()

        # we don't have third dimension of range view
        return rv, label_rv, numpy.stack((proj_x, proj_y), axis=1), cover_map

    def do_range_project(self, points, proj_W, proj_H):
        fov_up = self.proj_fov_up / 180.0 * numpy.pi  # field of view up in rad
        fov_down = self.proj_fov_down / 180.0 * numpy.pi  # field of view down in rad
        fov = abs(fov_down) + abs(fov_up)  # get field of view total in rad

        # get depth of all points
        depth = numpy.linalg.norm(points, 2, axis=1)

        # get scan components
        scan_x = points[:, 0]
        scan_y = points[:, 1]
        scan_z = points[:, 2]

        # get angles of all points
        yaw = -numpy.arctan2(scan_y, scan_x)
        pitch = numpy.arcsin(scan_z / depth)

        # get projections in image coords
        proj_x = 0.5 * (yaw / numpy.pi + 1.0)  # in [0.0, 1.0]
        proj_y = 1.0 - (pitch + abs(fov_down)) / fov  # in [0.0, 1.0]

        # scale to image size using angular resolution
        proj_x *= proj_W  # in [0.0, W]
        proj_y *= proj_H  # in [0.0, H]

        # round and clamp for use as index
        proj_x = numpy.floor(proj_x)
        proj_x = numpy.minimum(proj_W - 1, proj_x)
        proj_x = numpy.maximum(0, proj_x).astype(numpy.int32)  # in [0,W-1]
        # proj_x = numpy.copy(proj_x)  # store a copy in orig order

        proj_y = numpy.floor(proj_y)
        proj_y = numpy.minimum(proj_H - 1, proj_y)
        proj_y = numpy.maximum(0, proj_y).astype(numpy.int32)  # in [0,H-1]
        # self.proj_y = numpy.copy(proj_y)  # stope a copy in original order
        return proj_y, proj_x, proj_H, proj_W
