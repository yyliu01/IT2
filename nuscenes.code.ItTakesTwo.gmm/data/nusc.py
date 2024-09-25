import os
from typing import Dict, List, Tuple

import numpy as np
from nuscenes import NuScenes

from data.label_mapping import NuscLidarSegLabelMappings
from data.lidar import LidarPointCloud, LidarSegDatabaseInterface
from data.utils import NuscenesLidarMethods, get_range_proj_coordinates


class NuscLidarSegDatabase(LidarSegDatabaseInterface):

    def __init__(self,
                 nusc_db: NuScenes,
                 label_mapping_name: str,
                 min_distance: float = 0.9,
                 max_distance: float = 100.0,
                 rings: int = 32,
                 split: str = "train",
                 partition: int = 10,
                 mode: str = "labelled",
                 path: str = "./"):

        self.db = nusc_db
        self._raw_mapping = NuscLidarSegLabelMappings.raw_mappings(nusc_db)[label_mapping_name]
        self.split = split
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.n_rings = rings
        assert self.min_distance >= 0.0, f'min_distance {min_distance} should not be negative.'
        assert self.max_distance > 0.0, f'max_distance {max_distance} should not be 0 or negative. '

        self.globalid2localid = {}
        for _id, _name in self.db.lidarseg_idx2name_mapping.items():
            self.globalid2localid[_id] = self.local2id[self.global2local[_name]]

        if split == "train":
            file_list = os.path.join(path, 'data', 'splits', f"{partition}pct", f"{mode}_list.txt")
            self._tokens = [line.rstrip() for line in tuple(open(file_list, "r"))]
        elif split == "valid":
            file_list = os.path.join(path, 'data', 'splits', f"{split}_list.txt")
            self._tokens = [line.rstrip() for line in tuple(open(file_list, "r"))]

        # for debug purpose only
        elif split == "mini_train":
            file_list = os.path.join(path, 'data', 'mini_splits', "666pct", f"{mode}_list.txt")
            self._tokens = [line.rstrip() for line in tuple(open(file_list, "r"))]

        elif split == "mini_valid":
            file_list = os.path.join(path, 'data', 'mini_splits', f"{split}_list.txt")
            self._tokens = [line.rstrip() for line in tuple(open(file_list, "r"))]

        else:
            raise FileNotFoundError

    @property
    def db_version(self) -> str:
        return self.db.version

    @property
    def tokens(self) -> List[str]:
        return self._tokens

    # @property
    # def tokens(self) -> List[str]:
    #
    #     if self._tokens is None:
    #
    #         sample_tokens = NuscenesLidarMethods.splits(split=self.split, db=self.db)
    #
    #         if self.data_split_list_path:
    #
    #             with open(self.data_split_list_path, "r") as f:
    #                 self.token_list_labeled = f.read().splitlines()
    #             print("Loading '{}' labeled samples ('{:.0f}%') from nuScenes under '{}' splits ...".format(
    #                 len(self.token_list_labeled), (len(self.token_list_labeled) / len(sample_tokens)) * 100, self.split)
    #             )
    #
    #             self.token_list_unlabeled = [i for i in sample_tokens if i not in self.token_list_labeled]
    #
    #             print("Loading '{}' unlabeled samples ('{:.0f}%') from nuScenes under '{}' splits ...".format(
    #                 len(self.token_list_unlabeled), (len(self.token_list_unlabeled) / len(sample_tokens)) * 100,
    #                 self.split)
    #             )
    #
    #             self.token_list_labeled = self.token_list_labeled * int(
    #                 np.ceil(len(self.token_list_unlabeled) / len(self.token_list_labeled)))
    #
    #             if self.if_sup_only:
    #                 self._tokens = self.token_list_labeled
    #             else:
    #                 self._tokens = self.token_list_unlabeled
    #
    #         else:
    #             self._tokens = sample_tokens
    #
    #     return self._tokens

    def load_from_db(self, token: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        pc = load_pointcloud_from_db(self.db, token, include_ring=True)
        points = pc.points[:-1].T

        sample = self.db.get('sample', token)
        sample_data_token = sample['data']['LIDAR_TOP']

        if 'test' in self.split:
            if self.split == 'train_test':
                lidarseg_labels_filename = '' + token
                points_label = np.fromfile(lidarseg_labels_filename, dtype=np.uint8)
            else:
                points_label = np.zeros((points.shape[0],), dtype=np.uint8)  # dummy labels; won't be needed in test
        else:
            lidarseg_labels_filename = os.path.join(
                self.db.dataroot,
                self.db.get('lidarseg', sample_data_token)['filename']
            )
            points_label = np.fromfile(lidarseg_labels_filename, dtype=np.uint8)

            assert len(points_label) == len(points), \
                f'lidar seg labels {len(points_label)} does not match lidar points {len(points)}'

            points_label = list(map(lambda x: self.globalid2localid.get(x), list(points_label)))

        depth = np.linalg.norm(points[:, :2], axis=1)
        points = points[depth >= self.min_distance]  # [n, 5]
        points_label = np.atleast_2d(points_label)[np.atleast_2d(depth) >= self.min_distance]  # [n, 5]
        return points[:, :4], points_label, points[:, 4]

    @property
    def caching_mode(self) -> bool:
        return False

    def load_range_view_coordinates(
            self,
            points: np.ndarray,
            proj_type: str = 'cylindrical',
            ring: np.ndarray = None,
            horiz_angular_res: float = 0.1875
    ) -> Tuple[np.ndarray, np.ndarray, int, int]:
        if proj_type == 'cylindrical' and ring is not None:
            proj_y, proj_x, ht, wt = get_range_proj_coordinates(
                points[:, :3],
                ring=ring,
                horiz_angular_res=horiz_angular_res,
                num_rings=self.n_rings
            )
        elif proj_type == 'spherical':
            proj_y, proj_x, ht, wt = get_range_proj_coordinates(
                points[:, :3],
                ring=None,
                horiz_angular_res=horiz_angular_res,
                num_rings=self.n_rings,
                proj_fov_up=10,
                proj_fov_down=-30
            )
        else:
            raise NotImplementedError

        return proj_y, proj_x, ht, wt

    @property
    def global2local(self) -> Dict[str, str]:
        return NuscenesLidarMethods.render_global2local(self.db, self._raw_mapping['global2local'])

    @property
    def local2id(self) -> Dict[str, int]:
        return self._raw_mapping['local2id']

    @property
    def raw_mapping(self) -> dict:
        return self._raw_mapping

    @property
    def xrange(self) -> Tuple[float, float]:
        return -60, 60

    @property
    def yrange(self) -> Tuple[float, float]:
        return -60, 60


def load_pointcloud_from_db(
        nuscenes: NuScenes,
        token: str,
        include_ring: bool = False,
        use_intensity: bool = True
) -> LidarPointCloud:
    pc = load_pointcloud_from_db_ref_channel(
        nuscenes,
        token,
        include_ring,
        use_intensity,
        channel="LIDAR_TOP",
    )
    return pc


def load_pointcloud_from_db_ref_channel(
        nuscenes: NuScenes,
        token: str,
        include_ring: bool = False,
        use_intensity: bool = True,
        channel: str = "LIDAR_TOP",
) -> LidarPointCloud:
    db = nuscenes

    sample_rec = db.get('sample', token)
    sample_data_token = sample_rec['data'][channel]
    sd_rec = db.get('sample_data', sample_data_token)

    pc = LidarPointCloud.from_file(os.path.join(db.dataroot, sd_rec['filename']))  # [6, m]

    timevector = np.zeros((1, pc.nbr_points()), dtype=np.float32)
    pc.points = np.concatenate((pc.points, timevector), axis=0)

    decoration_index = [0, 1, 2]  # always use [x, y, z]
    if use_intensity:
        decoration_index += [3]  # default use intensity
    if include_ring:
        decoration_index += [4]  # default exclude ring

    decoration_index += [-1]  # always add time vector; the last element of point cloud

    pc.points = pc.points[np.array(decoration_index)]  # [6, m]: x, y, z, intensity, ring, timestamp
    pc.points = np.asfortranarray(pc.points)  # [6, m]

    return pc
