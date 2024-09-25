import os
import time

import matplotlib.pyplot as plt
import numpy
import torch
from tqdm import tqdm

from utils.mixes.mix_op import CutMix


class Trainer:
    def __init__(self, hyp_param, vis_tool, metrics, criterion):
        super(Trainer, self).__init__()
        self.param = hyp_param
        self.tensorboard = vis_tool
        self.metrics = metrics
        self.criterion = criterion
        self.modality_cutmix = CutMix(self.param)

        from trainer.prototype import PrototypeGenerator
        self.prototype_enumerator = PrototypeGenerator(hyp_param)

    def train_epoch(self, epoch, data_loader, model, optimiser):
        model.train()
        self.metrics.reset()
        progress_bar = tqdm(range(self.param.iters_per_epoch), ncols=150, disable=self.param.local_rank > 0,
                            colour="red")

        for batch_idx in progress_bar:
            labelled_batch, unlabelled_batch = next(data_loader)
            # labelled batch information
            labelled_batch['point_feature'] = labelled_batch['point_feature'].cuda(non_blocking=True)
            labelled_batch['range_feature'] = labelled_batch['range_feature'].cuda(non_blocking=True)
            labelled_batch['point_coord'] = labelled_batch['point_coord'].cuda(non_blocking=True)
            labelled_batch['voxel_label'] = labelled_batch['voxel_label'].cuda(non_blocking=True)
            labelled_batch['range_label'] = labelled_batch['range_label'].cuda(non_blocking=True)
            labelled_batch['project_coord'] = labelled_batch['project_coord'].cuda(non_blocking=True)

            # unlabelled batch information
            unlabelled_batch['point_feature'] = unlabelled_batch['point_feature'].cuda(non_blocking=True)
            unlabelled_batch['range_feature'] = unlabelled_batch['range_feature'].cuda(non_blocking=True)
            unlabelled_batch['project_coord'] = unlabelled_batch['project_coord'].cuda(non_blocking=True)
            unlabelled_batch['point_coord'] = unlabelled_batch['point_coord'].cuda(non_blocking=True)
            unlabelled_batch['project_coord'] = unlabelled_batch['project_coord'].cuda(non_blocking=True)

            # produce pseudo logits and embeddings
            (original_voxel_logit, pseudo_range_logit_1), (pseudo_voxel_embed_1, _) = \
                model.module.produce_pseudo_labels(unlabelled_batch, branch_name='voxel', embedding=True)

            (original_range_logit, pseudo_voxel_logit_2), (pseudo_range_embed_2, _) = \
                model.module.produce_pseudo_labels(unlabelled_batch, branch_name='range', embedding=True)

            # implement posterior mixes
            pseudo_range_conf_1, pseudo_range_label_1 = torch.max(torch.softmax(pseudo_range_logit_1, dim=1), dim=1)
            pseudo_voxel_conf_2, pseudo_voxel_label_2 = torch.max(torch.softmax(pseudo_voxel_logit_2, dim=1), dim=1)
            mix_range_input_1, mix_range_coord_1, pseudo_range_label_1, pseudo_range_embed_2 = \
                self.modality_cutmix(unlabelled_batch['range_feature'], unlabelled_batch['project_coord'],
                                     pseudo_range_embed_2, pseudo_range_label_1, modality='range')
            unlabelled_batch.update({"range_feature": mix_range_input_1, "project_coord": mix_range_coord_1})
            mix_voxel_input_2, mix_voxel_coord_2, pseudo_voxel_label_2 = \
                self.modality_cutmix(unlabelled_batch['point_feature'], unlabelled_batch['point_coord'],
                                     pseudo_voxel_label_2,
                                     modality='voxel')

            unlabelled_batch.update({"point_feature": mix_voxel_input_2, "point_coord": mix_voxel_coord_2})

            """
            prediction
            """
            # for modality with id 1
            labelled_logit_1, labelled_embeds_1 = model(labelled_batch, branch_name='voxel')
            unlabelled_logit_1, unlabelled_embeds_1 = model(unlabelled_batch, branch_name='voxel')

            sup_loss_1 = self.criterion['voxel'](labelled_logit_1, labelled_batch['voxel_label'])
            unsup_loss_1 = self.criterion['voxel'](unlabelled_logit_1, pseudo_voxel_label_2)

            # indices voxel modality labels
            voxel_index_l = labelled_batch['point_coord']
            voxel_index_u = unlabelled_batch['point_coord']
            anchor_voxel, contras_voxel, predict_voxel, confidence_voxel, label_voxel = \
                model.module.indices_variables(labelled_embeds_1, unlabelled_embeds_1, pseudo_voxel_embed_1,
                                               torch.argmax(labelled_logit_1, dim=1),
                                               torch.argmax(unlabelled_logit_1, dim=1), labelled_batch['voxel_label'],
                                               pseudo_voxel_label_2, pseudo_voxel_conf_2, voxel_index_l, voxel_index_u,
                                               modality="voxel")
            prototype_voxel = self.prototype_enumerator.sampling(label_voxel)
            embedd_loss_1 = self.criterion['cross-modality'](predict_voxel, anchor_voxel, label_voxel, prototype_voxel,
                                                             confidence_voxel)
            curr_loss_1 = sup_loss_1 + unsup_loss_1 * self.param.unlabelled_weight + embedd_loss_1

            # update gradient
            optimiser["voxel"].zero_grad()
            curr_loss_1.backward()
            optimiser["voxel"].step()

            # for modality with id 2
            labelled_logit_2, labelled_embeds_2 = model(labelled_batch, branch_name='range')
            unlabelled_logit_2, unlabelled_embeds_2 = model(unlabelled_batch, branch_name='range')
            sup_loss_2 = self.criterion['range'](labelled_logit_2, labelled_batch['range_label'])
            unsup_loss_2 = self.criterion['range'](unlabelled_logit_2, pseudo_range_label_1)
            range_index_l = labelled_batch['project_coord']
            range_index_u = unlabelled_batch['project_coord']
            anchor_range, contras_range, predict_range, confidence_range, label_range = \
                model.module.indices_variables(labelled_embeds_2, unlabelled_embeds_2, pseudo_range_embed_2,
                                               torch.argmax(labelled_logit_2, dim=1),
                                               torch.argmax(unlabelled_logit_2, dim=1), labelled_batch['range_label'],
                                               pseudo_range_label_1, pseudo_range_conf_1, range_index_l, range_index_u,
                                               modality="range")
            prototype_range = self.prototype_enumerator.sampling(label_range)
            embedd_loss_2 = self.criterion['cross-modality'](predict_range, anchor_range, label_range, prototype_range,
                                                             confidence_range)
            curr_loss_2 = sup_loss_2 + unsup_loss_2 * self.param.unlabelled_weight + embedd_loss_2

            # update gradient
            optimiser["range"].zero_grad()
            curr_loss_2.backward()
            optimiser["range"].step()

            # update gmm clusters
            self.prototype_enumerator.fitting(training_samples=torch.cat([contras_voxel, contras_range]),
                                              training_labels=torch.cat([label_voxel, label_range]),
                                              confidence=torch.cat([confidence_voxel, confidence_range]))

            curr_idx = self.param.iters_per_epoch * epoch + batch_idx
            # update the learning rate based on poly decay
            curr_lr = self.param.lr * (1 - curr_idx / (self.param.iters_per_epoch * self.param.epochs)) ** 0.9
            for i in range(len(optimiser["voxel"].param_groups)):
                 optimiser["voxel"].param_groups[i]['lr'] = curr_lr
            for i in range(len(optimiser["range"].param_groups)):
                optimiser["range"].param_groups[i]['lr'] = curr_lr

            # print out
            pbar_txt = [epoch, sup_loss_1.item(), unsup_loss_1.item(), embedd_loss_1.item(),
                        sup_loss_2.item(), unsup_loss_2.item(), embedd_loss_2.item()]

            progress_bar.set_description("epoch {} | modality_1 sup {:.3f} unsup {:.3f} embed {:.3f} | "
                                         "modality_2 sup {:.3f} unsup {:.3f} embed {:.3f} ".format(*pbar_txt))

            if self.param.local_rank <= 0:
                self.tensorboard.upload_wandb_info({"train/voxel/supervised_loss": sup_loss_1.item(),
                                                    "train/voxel/unsupervised_loss": unsup_loss_1.item(),
                                                    "train/voxel/embedding_loss": embedd_loss_1.item(),
                                                    "train/voxel/lr": optimiser["voxel"].param_groups[0]['lr'],
                                                    "train/range/supervised_loss": sup_loss_2.item(),
                                                    "train/range/unsupervised_loss": unsup_loss_2.item(),
                                                    "train/range/embedding_loss": embedd_loss_2.item(),
                                                    "train/range/lr": optimiser["range"].param_groups[0]['lr']})

            del curr_loss_1, sup_loss_1, unsup_loss_1, embedd_loss_1
            del curr_loss_2, sup_loss_2, unsup_loss_2, embedd_loss_2

            for key, value in labelled_batch.items(): labelled_batch[key] = value.cpu()
            for key, value in unlabelled_batch.items(): unlabelled_batch[key] = value.cpu()

            del pseudo_range_label_1, pseudo_voxel_label_2
            del pseudo_voxel_embed_1, pseudo_range_embed_2
            del pseudo_range_conf_1, pseudo_voxel_conf_2

            del labelled_embeds_1, labelled_embeds_2, unlabelled_embeds_1, unlabelled_embeds_2

    @torch.no_grad()
    def validate_epoch(self, epoch, data_loader, model, branch_name):
        model.eval()
        self.metrics.reset()
        progress_bar = tqdm(range(len(data_loader)), ncols=135, colour="yellow")
        for batch_idx in progress_bar:
            val_info = next(data_loader)
            for key, value in val_info.items():
                if torch.is_tensor(value):
                    val_info[key] = value.cuda(non_blocking=True)

            (current_view_predict, other_view_predict), _ = \
                model.module.produce_pseudo_labels(val_info, branch_name=branch_name)
            current_view_predict = torch.argmax(current_view_predict, dim=1)
            other_view_predict = torch.argmax(other_view_predict, dim=1)
            self.metrics.valid.add_batch(current_view_predict, val_info, branch_name)

            curr_acc = self.metrics.valid.get_acc()
            curr_iou, class_iou = self.metrics.valid.get_iou()
            progress_bar.set_description("epoch {} | modality {} | iou {} acc {} "
                                         "|".format(epoch, branch_name, curr_iou, curr_acc))

            for key, value in val_info.items():
                if torch.is_tensor(value):
                    val_info[key] = value.cpu()

            if self.param.local_rank <= 0 and batch_idx == 0:
                self.tensorboard.upload_scenes(current_view_predict, other_view_predict,
                                               val_info, branch_name_=branch_name)

        self.tensorboard.upload_class_iou_bar(class_iou, branch_name_=branch_name)
        self.tensorboard.upload_wandb_info({"val/{}/iou".format(branch_name): curr_iou.item(),
                                            "val/{}/acc".format(branch_name): curr_acc.item()})
        del val_info
        return numpy.round(curr_iou.item(), 4)

    def save_ckpts(self, epoch, model, optimiser, name):
        state = {
            'arch': type(model).__name__,
            'epoch': epoch,
            'state_dict': model.module.state_dict(),
            'optimiser': optimiser,
            'config': self.param
        }
        ckpt_name = '{}_model_e{}.pth'.format(str(name), str(epoch))
        filename = os.path.join(self.param.ckpts_dir, ckpt_name)
        print('\nSaving a checkpoint to {} ...'.format(str(filename)))
        torch.save(state, filename)
