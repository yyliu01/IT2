import os
import random
import argparse

import numpy
import torch
import datetime
from easydict import EasyDict
from itertools import cycle
from data.semantic_kitti import SemanticKittiDatabase
from modality.network import Models
from trainer.train import Trainer
from utils.evaluation.ioueval import Metrics
from utils.losses.range_loss import RangeModalityLoss
from utils.losses.voxel_loss import VoxelModalityLoss
from utils.tensorboard import Tensorboard


def seed_it(seed):
    random.seed(seed)
    os.environ["PYTHONSEED"] = str(seed)
    numpy.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.manual_seed(seed)


def main(local_rank, nprocess, hyp_param):
    if hyp_param.ddp:
        hyp_param.local_rank = local_rank
        torch.distributed.init_process_group(backend='nccl', init_method='env://', rank=hyp_param.local_rank,
                                             world_size=hyp_param.gpus, timeout=datetime.timedelta(seconds=5400))

    if hyp_param.local_rank <= 0:
        vis_tool = Tensorboard(hyp_param_=hyp_param, online=hyp_param.wandb_online)
    else:
        vis_tool = None

    seed_it(hyp_param.seed + hyp_param.local_rank)

    # produce the dataset
    labelled_dataset = SemanticKittiDatabase(root=hyp_param.root_dir, data_dir=hyp_param.data_path,
                                             partition=hyp_param.labelled_percent,
                                             range_info=(hyp_param.fov_up, hyp_param.fov_down, hyp_param.rings,
                                                         hyp_param.horizontal_angular_resolution,
                                                         hyp_param.ignore_index),
                                             voxel_info=(hyp_param.max_volume_bound, hyp_param.min_volume_bound,
                                                         hyp_param.intervals, hyp_param.grid_size,
                                                         hyp_param.ignore_index),
                                             split="train", mode="labelled", augment=True)

    unlabelled_dataset = SemanticKittiDatabase(root=hyp_param.root_dir, data_dir=hyp_param.data_path,
                                               partition=hyp_param.labelled_percent,
                                               range_info=(hyp_param.fov_up, hyp_param.fov_down, hyp_param.rings,
                                                           hyp_param.horizontal_angular_resolution,
                                                           hyp_param.ignore_index),
                                               voxel_info=(hyp_param.max_volume_bound, hyp_param.min_volume_bound,
                                                           hyp_param.intervals, hyp_param.grid_size,
                                                           hyp_param.ignore_index),
                                               split="train", mode="unlabelled", augment=True)

    validation_dataset = SemanticKittiDatabase(root=hyp_param.root_dir, data_dir=hyp_param.data_path,
                                               partition=hyp_param.labelled_percent,
                                               range_info=(hyp_param.fov_up, hyp_param.fov_down, hyp_param.rings,
                                                           hyp_param.horizontal_angular_resolution,
                                                           hyp_param.ignore_index),
                                               voxel_info=(hyp_param.max_volume_bound, hyp_param.min_volume_bound,
                                                           hyp_param.intervals, hyp_param.grid_size,
                                                           hyp_param.ignore_index),
                                               split="valid", mode="nil", augment=False)

    if hyp_param.ddp:
        labelled_sampler = torch.utils.data.distributed.DistributedSampler(labelled_dataset)
        unlabelled_sampler = torch.utils.data.distributed.DistributedSampler(unlabelled_dataset)
    else:
        labelled_sampler = torch.utils.data.RandomSampler(labelled_dataset)
        unlabelled_sampler = torch.utils.data.RandomSampler(unlabelled_dataset)

    labelled_loader = torch.utils.data.DataLoader(labelled_dataset,
                                                  batch_size=hyp_param.batch_size,
                                                  sampler=labelled_sampler,
                                                  num_workers=hyp_param.num_workers,
                                                  collate_fn=labelled_dataset.collate_batch,
                                                  drop_last=True)

    unlabelled_loader = torch.utils.data.DataLoader(unlabelled_dataset,
                                                    batch_size=hyp_param.batch_size,
                                                    sampler=unlabelled_sampler,
                                                    num_workers=hyp_param.num_workers,
                                                    collate_fn=unlabelled_dataset.collate_batch,
                                                    drop_last=True)

    val_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=hyp_param.batch_size,
                                             num_workers=hyp_param.num_workers,
                                             collate_fn=validation_dataset.collate_batch,
                                             shuffle=False, drop_last=False)

    # update the iterations for each of the epoch
    hyp_param.iters_per_epoch = len(unlabelled_loader)
    hyp_param.upload_image_step = int(hyp_param.iters_per_epoch / 2)

    model = Models(param=hyp_param)
    optimiser = {"voxel": torch.optim.AdamW(model.branches['voxel'].parameters(), lr=hyp_param.lr,
                                            betas=hyp_param.betas, weight_decay=hyp_param.weight_decay),
                 "range": torch.optim.AdamW(model.branches['range'].parameters(), lr=hyp_param.lr,
                                            betas=hyp_param.betas, weight_decay=hyp_param.weight_decay)}

    criterion = {"voxel": VoxelModalityLoss(ignore_index=hyp_param.ignore_index,
                                            modality="voxel"),
                 "range": RangeModalityLoss(alpha=[1, 1.5, 1],
                                            ignore_index=hyp_param.ignore_index)}

    if hyp_param.ddp:
        # for multi-GPUs DDP training
        torch.cuda.set_device(hyp_param.local_rank)
        model.cuda(hyp_param.local_rank)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[hyp_param.local_rank],
                                                          find_unused_parameters=True)
    else:
        # for single GPU training
        model = torch.nn.DataParallel(model, device_ids=[0])

    trainer = Trainer(hyp_param=hyp_param, vis_tool=vis_tool,
                      metrics=Metrics(n_classes=hyp_param.n_classes, ignore_index=hyp_param.ignore_index,
                                      device="cuda"),
                      criterion=criterion)

    start_epoch = 0

    for epoch in range(start_epoch, hyp_param.epochs):
        # set sampler for ddp training
        if hyp_param.ddp:
            labelled_sampler.set_epoch(epoch)
            unlabelled_sampler.set_epoch(epoch)

        trainer.train_epoch(epoch=epoch, data_loader=iter(zip(cycle(labelled_loader), unlabelled_loader)),
                            model=model, optimiser=optimiser)

        if hyp_param.local_rank <= 0 and (epoch % 5 == 0 or epoch > 30):

            curr_voxel_iou = trainer.validate_epoch(epoch=epoch, branch_name="voxel", data_loader=iter(val_loader),
                                                    model=model)
            curr_range_iou = trainer.validate_epoch(epoch=epoch, branch_name="range", data_loader=iter(val_loader),
                                                    model=model)

            trainer.save_ckpts(epoch, model, optimiser, name='voxel={}_range={}.pth'.format(str(curr_voxel_iou),
                                                                                            str(curr_range_iou)))

        if hyp_param.gpus > 1:
            torch.distributed.barrier()

    if hyp_param.local_rank <= 0:
        vis_tool.finish()
        print('finish.')

    return


if __name__ == '__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='Lidar Semi-supervised Segmentation')

    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N')

    parser.add_argument("--local_rank", type=int, default=-1, help='multi-process training for DDP')
    parser.add_argument('-g', '--gpus', default=1, type=int, help='number of gpus per node')

    # training hyp-value settings
    parser.add_argument('--batch_size', default=2, type=int)

    parser.add_argument('--epochs', default=50, type=int,
                        help="total epochs that used for the training")

    parser.add_argument('--labelled_percent', default=10, type=int,
                        help="the labelled cases in the semi-supervised learning")

    parser.add_argument('--weight', default=1., type=float,
                        help='default weight')

    parser.add_argument('--lr', default=8e-3, type=float,
                        help='default learning rate')

    parser.add_argument("--unlabelled_weight", type=float, default=1.,
                        help='the weight for unlabelled data')

    # utilise dgx-pvc or not
    # parser.add_argument('--pvc', action="store_true")

    # log & visualization settings
    # parser.add_argument('--online', action="store_true",
    #                     help='switch on for visualization; switch off for debug')

    """
    please note: we fix the *uniform type* in here for easy organising the code. 
    Feel free to change the sample id, where the 
    partial sampling lists are in: https://github.com/llijiang/GuidedContrast.
    significant sampling lists are in: https://github.com/l1997i/LiM3D.
    and also remember to change the grid_size to be (480, 360, 32) in configs/config.py file.
    
    # parser.add_argument('--sample', type=str, default='uniform', help='the sample type for training.')
    """

    args = parser.parse_args()
    args.ddp = True if args.gpus > 1 else False

    from configs.config import C

    args = EasyDict({**C, **vars(args)})

    """
    update for fine-tuning phase
    """

    if args.ddp:
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '9901'
        torch.multiprocessing.spawn(main, nprocs=args.gpus, args=(args.gpus, args))
    else:
        main(-1, 1, args)
