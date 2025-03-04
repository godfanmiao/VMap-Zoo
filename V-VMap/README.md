<div align="center">
  <h1>V-VMAP: 车载单目相机实时生成矢量地图的关键技术</h1>
</div>

## Introduction
本项目中的网络采用全卷积结构替代Transformer结构，提高推理效率

## Deployment
### 1. Environment
**Step 1.** Create conda environment and activate it.

```
conda create -n vvmap python=3.8
conda activate vvmap
```

**Step 2.** Install PyTorch.

```
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```

**Step 3.** Install MMCV series.

```
# Install mmcv-series
pip install mmcv-full==1.6.0
pip install mmdet==2.28.2
pip install mmsegmentation==0.30.0
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
git checkout v1.0.0rc6 
pip install -e .
```

**Step 4.** Install other requirements.

```
pip install -r requirements.txt
```

### 2. Data Preparation
**Step 1.** Download [NuScenes](https://www.nuscenes.org/download) dataset to `./datasets/nuScenes`.

**Step 2.** Download [Argoverse2 (sensor)](https://argoverse.github.io/user-guide/getting_started.html#download-the-datasets) dataset to `./datasets/av2`.

**Step 3.** Generate annotation files for NuScenes dataset.

```
python tools/nuscenes_converter.py --data-root ./datasets/nuscenes
```

**Step 4.** Generate annotation files for Argoverse2 dataset.

```
python tools/argoverse_converter.py --data-root ./datasets/av2
```

### 3. Training and Validating
To train a model with 2 GPUs:

```
bash tools/dist_train.sh ${CONFIG} 2
```

To validate a model with 8 GPUs:

```
bash tools/dist_test.sh ${CONFIG} ${CEHCKPOINT} 2 --eval
```

### Results on NuScense

| Ground Truth | Pred Result |
|--------------|-------------|
|![0_gt.png](resources%2F0_gt.png)              |![0_pred.png](resources%2F0_pred.png)             |
|![1_gt.png](resources%2F1_gt.png)              |![1_pred.png](resources%2F1_pred.png)             |
|![2_gt.png](resources%2F2_gt.png)              |![2_pred.png](resources%2F2_pred.png)             |
|![3_gt.png](resources%2F3_gt.png)              |![3_pred.png](resources%2F3_pred.png)             |
|![4_gt.png](resources%2F4_gt.png)              |![4_pred.png](resources%2F4_pred.png)             |
|![5_gt.png](resources%2F5_gt.png)              |![5_pred.png](resources%2F5_pred.png)             |
|![6_gt.png](resources%2F6_gt.png)              |![6_pred.png](resources%2F6_pred.png)             |
|![7_gt.png](resources%2F7_gt.png)              |![7_pred.png](resources%2F7_pred.png)             |

## Publications

## Patents