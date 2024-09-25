from abc import ABC
import torch
import torch.nn as nn


class ContrastLoss(nn.Module, ABC):
    def __init__(self, param):
        super(ContrastLoss, self).__init__()
        self.param = param
        self.embeds_num = param.embeds_num
        self.embeds_dim = param.embeds_dim
        self.temperature = param.temperature
        self.ignore_index = param.ignore_index

    def define_indices_num(self, num_hard_, num_easy_):
        if num_hard_ >= self.embeds_num / 2 and num_easy_ >= self.embeds_num / 2:
            num_hard_keep = self.embeds_num // 2
            num_easy_keep = self.embeds_num - num_hard_keep
        elif num_hard_ >= self.embeds_num / 2:
            num_easy_keep = num_easy_
            num_hard_keep = self.embeds_num - num_easy_keep
        elif num_easy_ >= self.embeds_num / 2:
            num_hard_keep = num_hard_
            num_easy_keep = self.embeds_num - num_hard_keep
        else:
            return num_hard_, num_easy_

        return num_hard_keep, num_easy_keep

    def extraction_samples(self, y_hat, z_hat, y, prototype_, confidence_):
        """
        note: y := (ground_truth, pseudo_label)
        """
        category_list = torch.unique(y)
        category_list = category_list[category_list != self.param.ignore_index]
        anchor_feat = []
        anchor_label = []
        prototype_feat = []
        fetched_confidence = []

        for cls_id in category_list:
            hard_indices = ((y == cls_id) & (y_hat != cls_id)).nonzero()
            easy_indices = ((y == cls_id) & (y_hat == cls_id)).nonzero()
            num_of_hard_sample, num_of_easy_sample = \
                self.define_indices_num(hard_indices.shape[0], easy_indices.shape[0])
            if (num_of_easy_sample + num_of_hard_sample) <= 1: continue
            hard_indices = hard_indices[torch.randperm(hard_indices.shape[0])][:num_of_hard_sample]
            easy_indices = easy_indices[torch.randperm(easy_indices.shape[0])][:num_of_easy_sample]
            total_indices = torch.cat([hard_indices, easy_indices], dim=0)
            anchor_feat.append(z_hat[total_indices, :])
            prototype_feat.append(prototype_[total_indices, :])
            anchor_label.append(y[total_indices])
            fetched_confidence.append(confidence_[total_indices])

        anchor_feat = torch.cat(anchor_feat).squeeze(1)
        anchor_label = torch.cat(anchor_label)
        prototype_feat = torch.cat(prototype_feat).squeeze(1)
        fetched_confidence = torch.cat(fetched_confidence)
        return anchor_feat, anchor_label, prototype_feat, anchor_label.detach(), fetched_confidence

    def forward(self, predict, embeds, labels, prototype, confidence=None, modality="voxel"):
        # print(predict.shape, embeds.shape, labels.shape, prototype.shape, confidence.shape)
        feats, labels, c_feats, c_labels, conf_ = \
            self.extraction_samples(predict, embeds, labels, prototype,
                                    confidence)

        loss = self.info_nce(feats, labels, c_feats, c_labels, conf_) \
            if feats.nelement() > 0 and c_feats.nelement() > 0 else feats.mean() * .0

        return loss

    def info_nce(self, feats_, labels_, c_feats_, c_labels_, conf_):
        # same class be positive; different classes be negative
        mask = torch.eq(labels_, torch.transpose(c_labels_, 0, 1)).float()
        anchor_dot_contrast = torch.div(torch.matmul(feats_, torch.transpose(c_feats_, 0, 1)),
                                        self.temperature)
        conf_ = torch.matmul(conf_, torch.transpose(conf_, 0, 1))
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        neg_mask = 1 - mask

        # avoid the self duplicate issue
        mask = mask.fill_diagonal_(0.)
        neg_logits = (torch.exp(logits) * neg_mask).sum(1, keepdim=True)
        # neg_logits = torch.clip(neg_logits, min=self.param.epsilon)

        # define logits => x
        # x [400x400] <- all_pairs and x=log(exp(x))
        # y [400x1] <- sum of all negative pairs
        # log_prob [400x400] -> each sample is a pair
        # log_prob -> log(exp(x))-log(exp(x) + exp(y))
        # log_prob -> log{exp(x)/[exp(x)+exp(y)]}
        log_prob = logits - torch.log(torch.exp(logits) + neg_logits)
        log_prob = log_prob * conf_
        loss = - (mask * log_prob).sum(1) / mask.sum(1)
        return loss.mean()

