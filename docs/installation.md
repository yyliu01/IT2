# Installation

The project is based on PyTorch 2.3.1 with Python 3.10. Our work was trained using 4xA100 GPUs.

## 1. Clone the Git  repo

``` shell
$ git clone https://github.com/yyliu01/IT2
$ cd IT2
```

## 2. Install dependencies

1) create conda env
    ```shell
    $ conda env create -f it2.yml
    ```
2) install the torch 2.3.1
    ```shell
    $ conda activate it2
    # IF cuda 11.8:
    $ pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118
    # IF cuda 12.1:
    $ pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
    ```
3) install torch-scatter
```shell
# you might encounter version issue, please see: https://pypi.org/project/torch-scatter/
pip install torch-scatter 
```

4) install spconv 
```shell
# please follow the guide on: https://github.com/traveller59/spconv
pip install spconv-cu102 # for CUDA 10.2
pip install spconv-cu113 # for CUDA 11.3 
pip install spconv-cu114 # for CUDA 11.4
pip install spconv-cu117 # for CUDA 11.7
pip install spconv-cu120 # for CUDA 12.0
```

## 3. Prepare dataset

### SemanticKITTI & ScribbleKITTI

1) please download semantickitti from the official website in [here](https://www.semantic-kitti.org/) and
2) download scribblekitti from [here](https://github.com/ouenal/scribblekitti).
3) specify their paths in **configs/config.py** file, which is **C.data_path**.
4) please note that, both of these two share same input scans but different labels in *training set*.
* (optionally) you can download them from my [google drive](https://drive.google.com/drive/folders/1-LiBxk01UvH00POIfeNZO69u16Lc1zwh?usp=sharing).


### nuScenes

1) you can download from the official website in [here](https://www.nuscenes.org/).
2) specify the nuscenes dataset path in **configs/config.py** file, which is **C.data_path**.

## 4. Dataset Structure
1). the tree structures of the nuscenes dataset are shown below.

```
nuscenes
├── lidarseg
│   ├── v1.0-mini
│   ├── v1.0-test
│   └── v1.0-trainval
├── maps
│   ├── 36092f0b03a857c6a3403e25b4b7aab3.png
│   ├── 37819e65e09e5547b8a3ceaefba56bb2.png
│   ├── 53992ee3023e5494b90c316c183be829.png
│   └── 93406b464a165eaba6d9de76ca09f5da.png
├── nuscenes_infos_test.pkl
├── nuscenes_infos_train.pkl
├── nuscenes_infos_val.pkl
├── samples
│   └── LIDAR_TOP
├── sweeps
│   └── LIDAR_TOP
├── v1.0-mini
│   ├── category.json
│   └── lidarseg.json
├── v1.0-test
│   ├── category.json
│   └── lidarseg.json
└── v1.0-trainval
    ├── attribute.json
    ├── calibrated_sensor.json
    ├── category.json
    ├── ego_pose.json
    ├── instance.json
    ├── lidarseg.json
    ├── log.json
    ├── map.json
    ├── sample_annotation.json
    ├── sample_data.json
    ├── sample.json
    ├── scene.json
    ├── sensor.json
    └── visibility.json
```
    
2) the tree structures of the kitti datasets are shown below.

```
KITTI/
├── kitti_input
│   ├── test
│   ├── train
│   └── valid
├── kitti_label
│   ├── test
│   ├── train
│   └── valid
└── scribble_label
    └── train
```

