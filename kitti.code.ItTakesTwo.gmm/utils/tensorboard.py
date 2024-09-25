import os
import numpy
import torch
import wandb

color_map = {"ignore": (0, 0, 0), "car": (245, 150, 100), "bicycle": (245, 230, 100), "motorcycle": (150, 60, 30), "truck": (180, 30, 80),
             "bus": (250, 80, 100), "person": (30, 30, 255), "bicyclist": (200, 40, 255), "motorcyclist": (90, 30, 150),
             "road": (255, 0, 255), "parking": (255, 150, 255), "sidewalk": (75, 0, 75), "other-ground": (75, 0, 175),
             "building": (0, 200, 255), "fence": (50, 120, 255), "vegetation": (0, 175, 0), "trunck": (0, 60, 135),
             "terrian": (80, 240, 150), "pole": (150, 240, 255), "traffic-sign": (0, 0, 255)}

acc_map = {"fail": (255, 0, 0), "success": (128, 128, 128), "ignore_label": (0, 0, 0)}


class Tensorboard:
    def __init__(self, hyp_param_, online=False):
        os.environ['WANDB_API_KEY'] = "6cde1f75fbdf236d8e89b77d313d74b40e3d8d5f"
        os.system("wandb login")
        os.system("wandb {}".format("online" if online else "disabled"))
        self.tensor_board = wandb.init(project=hyp_param_.project_name, name=hyp_param_.experiment_name,
                                       config=hyp_param_, settings=wandb.Settings(code_dir="."))
        # self.root_dir = os.path.join(hyp_param_.root_path, hyp_param_.experim_name)
        self.local_rank = hyp_param_.local_rank
        self.param = hyp_param_

    def upload_wandb_info(self, info_dict):
        for _, info in enumerate(info_dict):
            self.tensor_board.log({info: info_dict[info]})
        return

    def demonstrate_cylinder_acc(self, points_coord, points_predict, points_label, name):
        loc_ = points_coord.cpu().numpy()
        label_idx = points_label.squeeze().cpu().numpy()
        pred_idx = points_predict.squeeze().cpu().numpy()
        res_idx = numpy.zeros_like(label_idx)
        res_idx[label_idx == pred_idx] = 1

        # ignore
        res_idx[label_idx == self.param.ignore_index] = 2

        acc_map_ = numpy.asarray(list(acc_map.values()))
        acc = numpy.apply_along_axis(lambda x: acc_map_[x], 1, res_idx[:, numpy.newaxis]).squeeze()

        self.tensor_board.log({"accuracy/[cylinder_view] {}".format(name):
            wandb.Object3D({
                "type": "lidar/beta",
                "points": numpy.concatenate([loc_, acc], axis=1)})})

        return

    def demonstrate_range_acc(self, predict, mask, name):

        mask = mask.clone().detach().cpu().float()
        mask = torch.nn.functional.interpolate(mask.unsqueeze(1), (mask.shape[-2] * 10, mask.shape[-1]),
                                               mode='nearest').squeeze().long().unsqueeze(0)
        predict = predict.clone().detach().cpu().float()
        predict = torch.nn.functional.interpolate(predict.unsqueeze(1), (predict.shape[-2] * 10, predict.shape[-1]),
                                                  mode='nearest').squeeze().long().unsqueeze(0)
        res_idx = numpy.zeros_like(mask)
        res_idx[predict == mask] = 1
        # ignore
        res_idx[mask == self.param.ignore_index] = 2

        acc_map_ = numpy.asarray(list(acc_map.values()))
        acc = numpy.apply_along_axis(lambda x: acc_map_[x], 1, res_idx[:, numpy.newaxis]).squeeze(0)

        self.tensor_board.log(
            {"accuracy/[range_view] {}".format(name): [
                wandb.Image(j.transpose(1, 2, 0), caption="id {}".format(str(i)))
                for i, j in enumerate(acc)]})

        return

    def demonstrate_cylinder_view(self, points_coord, points_label, name):
        loc_ = points_coord.cpu().numpy()
        label_idx = points_label.squeeze().cpu().numpy()
        class_map_ = numpy.asarray(list(color_map.values()))
        label = numpy.apply_along_axis(lambda x: class_map_[x], 1, label_idx[:, numpy.newaxis]).squeeze()

        self.tensor_board.log({"visualisation/[cylinder_view] {}".format(name):
            wandb.Object3D({
                "type": "lidar/beta",
                "points": numpy.concatenate([loc_, label], axis=1)})})

        return

    def demonstrate_range_view(self, label, name='range'):
        label = label.clone().detach().cpu().float()
        label = torch.nn.functional.interpolate(label.unsqueeze(1), (label.shape[-2] * 10, label.shape[-1]),
                                                mode='nearest').squeeze().long().unsqueeze(0)

        class_map_ = numpy.asarray(list(color_map.values()))
        label = numpy.apply_along_axis(lambda x: class_map_[x], 1, label[:, numpy.newaxis]).squeeze(0)
        self.tensor_board.log(
            {"visualisation/[range_view] {}".format(name): [
                wandb.Image(j.transpose(1, 2, 0), caption="id {}".format(str(i)))
                for i, j in enumerate(label)]})

    def upload_scenes(self, current_view_predict_, other_view_predict_, batch_info_, branch_name_):
        """
        visualise 3 types of views:
            1). the results in cart coordinates (from 2 modalities)
            2). the results in range view
            3). the results in cylinder coordinates
        """

        # cart coordinates based on cylinder view
        # cart_coordinates = self.transfer_view_from_cylinder_to_cart(batch_info_['point_coord'])
        # upload ground truth
        # self.demonstrate_cylinder(cart_coordinates, batch_info_['point_label'], name='cylinder')

        cylinder_point_coord = batch_info_['point_coord']
        if branch_name_ == "voxel":
            current_view_predict_ = current_view_predict_[cylinder_point_coord[:, 0], cylinder_point_coord[:, 1],
                                                          cylinder_point_coord[:, 2], cylinder_point_coord[:, 3]]
            cylinder_point_coord_ = cylinder_point_coord[:, 1:]
            cylinder_point_label_ = batch_info_['point_label']
            for idx, offset_ in enumerate(batch_info_['offset']):
                start_ = batch_info_['offset'][idx - 1] if idx > 0 else 0
                instance_cylinder_point_coord_ = self.transfer_view_from_cylinder_to_cart(cylinder_point_coord)
                instance_cylinder_point_coord_ = instance_cylinder_point_coord_[start_:offset_]
                # instance_cylinder_point_coord_ = cylinder_point_coord_[start_:offset_]
                instance_cylinder_point_label_ = cylinder_point_label_[start_:offset_]
                instance_current_view_predict = current_view_predict_[start_:offset_]
                instance_other_view_predict = other_view_predict_[idx]
                instance_range_mask = batch_info_['range_label'][idx]

                # cylinder coordinate results
                self.demonstrate_cylinder_view(instance_cylinder_point_coord_, instance_current_view_predict,
                                               name="{}_modality".format(branch_name_))

                self.demonstrate_cylinder_view(instance_cylinder_point_coord_, instance_cylinder_point_label_,
                                               name="label")

                self.demonstrate_cylinder_acc(instance_cylinder_point_coord_,
                                              points_predict=instance_current_view_predict,
                                              points_label=instance_cylinder_point_label_,
                                              name="{}_modality".format(branch_name_))

                self.demonstrate_range_acc(instance_other_view_predict.unsqueeze(0),
                                           mask=instance_range_mask.unsqueeze(0),
                                           name="{}_modality".format(branch_name_))

                self.demonstrate_range_view(instance_other_view_predict.unsqueeze(0),
                                            name="{}_modality".format(branch_name_))

        else:
            other_view_predict_ = other_view_predict_[cylinder_point_coord[:, 0], cylinder_point_coord[:, 1],
                                                      cylinder_point_coord[:, 2], cylinder_point_coord[:, 3]]
            cylinder_point_coord_ = cylinder_point_coord[:, 1:]
            cylinder_point_label_ = batch_info_['point_label']
            for idx, offset_ in enumerate(batch_info_['offset']):
                start_ = batch_info_['offset'][idx - 1] if idx > 0 else 0
                instance_cylinder_point_coord_ = cylinder_point_coord_[start_:offset_]
                instance_cylinder_point_label_ = cylinder_point_label_[start_:offset_]
                instance_other_view_predict = other_view_predict_[start_:offset_]
                instance_current_view_predict = current_view_predict_[idx]
                instance_range_mask = batch_info_['range_label'][idx]

                # spherical 2d coordinate results
                self.demonstrate_range_view(instance_current_view_predict.unsqueeze(0),
                                            name="{}_modality".format(branch_name_))

                self.demonstrate_range_view(instance_range_mask.unsqueeze(0),
                                            name="label")

                self.demonstrate_range_acc(instance_current_view_predict.unsqueeze(0),
                                           mask=instance_range_mask.unsqueeze(0),
                                           name="{}_modality".format(branch_name_))

                self.demonstrate_cylinder_acc(instance_cylinder_point_coord_,
                                              points_predict=instance_other_view_predict,
                                              points_label=instance_cylinder_point_label_,
                                              name="{}_modality".format(branch_name_))

                self.demonstrate_cylinder_view(instance_cylinder_point_coord_, instance_other_view_predict,
                                               name="{}_modality".format(branch_name_))

        return

    def transfer_view_from_cylinder_to_cart(self, cylinder_coord):
        def polar2cart(input_xyz_polar):
            x = input_xyz_polar[:, 0] * numpy.cos(input_xyz_polar[:, 1])
            y = input_xyz_polar[:, 0] * numpy.sin(input_xyz_polar[:, 1])
            return numpy.stack((x, y, input_xyz_polar[:, 2]), axis=1)

        denormalise_coord = cylinder_coord[:, 1:] * self.param.intervals + self.param.min_volume_bound
        return torch.from_numpy(polar2cart(denormalise_coord))

    def upload_class_iou_bar(self, iou_bar, branch_name_):
        class_ = list(color_map.keys())
        collect = []
        for i in range(0, len(class_)):
            collect.append([class_[i], iou_bar[i]])
        table = wandb.Table(data=collect, columns=['class', 'iou'])
        self.tensor_board.log({"val/{}/class_iou".format(branch_name_): wandb.plot.bar(table, "class", "iou")})

    @staticmethod
    def watch(module):
        wandb.watch(models=module)

    @staticmethod
    def finish():
        wandb.finish()


"""

    def upload_3d_scenes(self, points, branch_id, status='train/', upload_img_number=8):

        # def polar2cat(input_xyz_polar):
        #     x = input_xyz_polar[:, 0] * numpy.cos(input_xyz_polar[:, 1]/numpy.max(input_xyz_polar))
        #     y = input_xyz_polar[:, 0] * numpy.sin(input_xyz_polar[:, 1]/numpy.max(input_xyz_polar))
        #     return numpy.stack((x, y, input_xyz_polar[:, 2]), axis=1)

        # the first dimension is batch_idx.
        loc_ = points['point_coord'][:, 1:].cpu().numpy()
        # loc_ = polar2cat(loc_)
        label_idx = points['point_label'].squeeze().cpu().numpy()
        pred_idx = points['predict'].squeeze().cpu().numpy()
        res_idx = numpy.zeros_like(label_idx)
        res_idx[label_idx == pred_idx] = 1
        class_map_ = numpy.asarray(list(color_map.values()))
        acc_map_ = numpy.asarray(list(acc_map.values()))
        label = numpy.apply_along_axis(lambda x: class_map_[x], 1, label_idx[:, numpy.newaxis]).squeeze()
        pred = numpy.apply_along_axis(lambda x: class_map_[x], 1, pred_idx[:, numpy.newaxis]).squeeze()
        residual = numpy.apply_along_axis(lambda x: acc_map_[x], 1, res_idx[:, numpy.newaxis]).squeeze()

        res_3d_infos = []
        pred_3d_infos = []
        label_3d_infos = []

        for idx, offset_ in enumerate(points['offset'][:upload_img_number]):
            start_ = points['offset'][idx - 1] if idx > 0 else 0
            label_3d_infos.append(wandb.Object3D({
                "type": "lidar/beta",
                "points": numpy.concatenate([loc_[start_:offset_],
                                             label[start_:offset_]], axis=1)}))
            pred_3d_infos.append(wandb.Object3D({
                "type": "lidar/beta",
                "points": numpy.concatenate([loc_[start_:offset_],
                                             pred[start_:offset_]], axis=1)}))
            res_3d_infos.append(wandb.Object3D({
                "type": "lidar/beta",
                "points": numpy.concatenate([loc_[start_:offset_],
                                             residual[start_:offset_]], axis=1)}))

        self.tensor_board.log({status + "/3d/modality_{}_label".format(branch_id): label_3d_infos})
        self.tensor_board.log({status + "/3d/modality_{}_predict".format(branch_id): pred_3d_infos})
        self.tensor_board.log({status + "/3d/modality_{}_accuracy".format(branch_id): res_3d_infos})
"""
