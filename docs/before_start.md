# Getting Started

we visualize our training details via wandb (https://wandb.ai/site).

## visualization

1) you'll need to login
   ```shell 
   $ wandb login
   ```
2) you'll need to copy & paste you API key in terminal
   ```shell
   $ https://wandb.ai/authorize
   ```
   or add the key to the "code/config/config.py" with
   ```shell
   C.wandb_key = ""
   ```

## training
our code is trained with 4 x nvidia a100 gpus, alternatively, you can downsize the range view's input resolution to lower the training requirements, but please note that the final results cannot be guaranteed to match ours.


for training, please find the training scripts in "scripts" directory.

```shell 
$ ./scripts/nuscenes_run.sh your_defined_labelled_num

$ ./scripts/kitti_run.sh your_defined_labelled_num
```

## checkpoints
checkpoints and training logs are in [this google drive link](https://drive.google.com/drive/folders/1bFxr4YxBGVmRTA-C_w0_VTYSvvuk07cc?usp=sharing).

## training details
some examples of training detail, please see [this wandb link](https://wandb.ai/pyedog1976/IT2?nw=nwuserpyedog1976).

In details, after clicking the run (e.g., [nuscenes_final_10_pct](https://wandb.ai/pyedog1976/IT2/runs/eyskbow9?nw=nwuserpyedog1976)), you can checkout:

1) <img src="https://user-images.githubusercontent.com/102338056/167979073-1c1b3144-8a72-4d8d-9084-31d7fdab3e9b.png" width="26" height="22"> overall information (e.g., training command line, hardware information and training time).
2) <img src="https://user-images.githubusercontent.com/102338056/167978940-8c1f3d79-d062-4e7b-b56e-30b97d273ae8.png" width="26" height="22"> training details (e.g., loss curves, validation results and visualization)
3) <img src="https://user-images.githubusercontent.com/102338056/167979238-4847430f-aa0b-483d-b735-8a10b43293a1.png" width="26" height="22"> output logs (well, sometimes might crash ...)