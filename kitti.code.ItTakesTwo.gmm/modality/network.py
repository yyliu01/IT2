import torch
import torch_scatter
from modality.range.fidnet.architecture import FIDNet
from modality.voxel.cylinder.architecture import Cylinder3D


class Models(torch.nn.Module):
    # modality_1 => voxel_view model: Cylinder3D
    # modality_2 => range_view model: FIDNet

    def __init__(self, param):
        super().__init__()
        self.param = param
        self.branches = torch.nn.ModuleDict()

        self.branches.update({"voxel": Cylinder3D(output_shape=param.grid_size,
                                                  nclasses=param.n_classes,
                                                  fea_dim=param.feat_dimension,
                                                  fea_compression=param.feat_compression,
                                                  out_pt_fea_dim=param.output_feature,
                                                  init_size=param.init_size, embedding_dim=param.embeds_dim)})

        self.branches.update({"range": FIDNet(num_cls=param.n_classes,
                                              embedding_dim=param.embeds_dim)})

    @staticmethod
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters())

    @torch.no_grad()
    def produce_pseudo_labels(self, batch, branch_name, embedding=True):
        current_view_logits, current_view_embeds = self.branches[branch_name](batch, embedding=embedding)
        other_view_logits = self.transfer_view(current_view_logits, batch["point_coord"],
                                               batch["project_coord"], branch_name)
        return (current_view_logits, other_view_logits), (current_view_embeds, torch.empty(1))

    def forward(self, batch_info, branch_name):
        logits, embeds = self.branches[branch_name](batch_info, embedding=True)
        return logits, embeds

    def transfer_view(self, pseudo_label_, point_coord_, project_coord_, branch_name_):
        assert branch_name_ in ['voxel', 'range'], 'unknown modality name in branch selection'
        return self.transfer_label_from_cylinder_to_range(pseudo_label_, point_coord_, project_coord_) \
            if branch_name_ == "voxel" else \
            self.transfer_label_from_range_to_cylinder(pseudo_label_, point_coord_, project_coord_)

    def transfer_label_from_range_to_cylinder(self, prediction, point_coord, project_coord):
        # transformation from range to point
        prediction = prediction[project_coord[:, 0], :, project_coord[:, 2], project_coord[:, 1]]
        # transformation from point to voxel
        voxel_prediction = torch.full(size=(self.param.batch_size, prediction.shape[1], *self.param.grid_size),
                                      fill_value=self.param.ignore_index, dtype=torch.float, device=prediction.device)
        voxel_prediction[point_coord[:, 0], :, point_coord[:, 1], point_coord[:, 2], point_coord[:, 3]] = prediction
        del prediction
        return voxel_prediction

    def transfer_label_from_cylinder_to_range(self, prediction, point_coord, project_coord):
        # transformation from volume to grid
        prediction = prediction[point_coord[:, 0], :, point_coord[:, 1], point_coord[:, 2], point_coord[:, 3]]
        # transformation from grid to range
        range_prediction = torch.full(size=(self.param.batch_size, prediction.shape[1], self.param.rings,
                                            self.param.horizontal_resolution),
                                      fill_value=self.param.ignore_index, dtype=torch.float, device=prediction.device)
        range_prediction[project_coord[:, 0], :, project_coord[:, 2], project_coord[:, 1]] = prediction
        del prediction
        return range_prediction

    @staticmethod
    def scatter_(x_, pos_):
        return torch_scatter.scatter_max(x_, pos_, dim=0)[0]

    def indices_voxel_coord(self, x, voxel_coord, unique_position_):
        x = x[voxel_coord[:, 0], voxel_coord[:, 1], voxel_coord[:, 2], voxel_coord[:, 3]]
        return self.scatter_(x, unique_position_)

    def indices_range_coord(self, x, proj_coord, unique_position_):
        x = x[proj_coord[:, 0], proj_coord[:, 2], proj_coord[:, 1]]
        return self.scatter_(x, unique_position_)

    def indices_variables(self, embed_l, embed_u, embed_u_other, predict_l, predict_u, label_l, label_u,
                          confidence, coord_l, coord_u, modality):

        _, unique_position_l = torch.unique(coord_l, dim=0, return_inverse=True)
        _, unique_position_u = torch.unique(coord_u, dim=0, return_inverse=True)
        embed_l = self.scatter_(embed_l, unique_position_l)
        embed_u = self.scatter_(embed_u, unique_position_u)
        embed_u_other = self.scatter_(embed_u_other, unique_position_u)

        indices_function = self.indices_voxel_coord if modality == "voxel" \
            else self.indices_range_coord

        predict_l = indices_function(predict_l, coord_l, unique_position_l)
        predict_u = indices_function(predict_u, coord_u, unique_position_u)

        label_l = indices_function(label_l, coord_l, unique_position_l)
        label_u = indices_function(label_u, coord_u, unique_position_u)

        confidence_l = torch.ones_like(label_l, device=label_l.device, dtype=torch.float)
        confidence_u = indices_function(confidence, coord_u, unique_position_u)

        # anchor-set, contrast-set, predict-set, confidence-set, label-set
        return torch.cat([embed_l, embed_u]), torch.cat([embed_l, embed_u_other]), \
            torch.cat([predict_l, predict_u]), torch.cat([confidence_l, confidence_u]), torch.cat([label_l, label_u])


