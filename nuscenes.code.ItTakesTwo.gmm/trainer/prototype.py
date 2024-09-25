import torch
import numpy
from trainer.gmm import GaussianMixture


class PrototypeGenerator(torch.nn.Module):
    def __init__(self, param):
        super(PrototypeGenerator, self).__init__()
        self.ignore_index = param.ignore_index
        self.voxel_size = numpy.prod(param.grid_size)
        self.range_size = param.horizontal_resolution * param.rings
        self.gmm = list(GaussianMixture(n_components=5, n_features=param.embeds_dim, covariance_type="diag",
                                        init_params="random").cuda() for _ in range(param.n_classes))
        self.param = param

    def sampling(self, sample_labels):
        samples = torch.zeros((*sample_labels.shape, self.param.embeds_dim), device=sample_labels.device)
        unique_pseudo_labels, counts = torch.unique(sample_labels, return_counts=True)
        for unique_id in torch.unique(unique_pseudo_labels):
            if unique_id == self.ignore_index: continue
            samples_mask = sample_labels == unique_id
            samples[samples_mask], _ = self.gmm[unique_id].sample(torch.sum(samples_mask).item())
        return samples

    def fitting(self, training_samples, training_labels, confidence):
        unique_pseudo_labels, counts = torch.unique(training_labels, return_counts=True)
        # just apply a super-small value for eliminate the potential noisy data.
        unique_pseudo_labels[counts <= 1] = self.ignore_index

        for unique_id in torch.unique(unique_pseudo_labels):
            if unique_id == self.ignore_index: continue
            # size: event_n * feature_dim
            current_sample = training_samples[training_labels == unique_id]
            self.gmm[unique_id].fit(current_sample, confidence[training_labels == unique_id], warm_start=True,
                                    n_iter=10)

