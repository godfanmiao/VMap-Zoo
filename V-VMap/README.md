<div align="center">
  <h1>V-VMAP: 车载单目相机实时生成矢量地图的关键技术</h1>
</div>

## Introduction
本工程实现了仅使用单目相机图像端到端生成视野范围内的矢量化HDMap，此外，本工程使用卷积替代Transformer结构，提升了推理效率

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

**Step 2.** Generate annotation files for NuScenes dataset.

```
python tools/nuscenes_converter.py --data-root ./datasets/nuscenes
```

### 3. Training and Validating
To train a model with 8 GPUs:

```
bash tools/dist_train.sh ${CONFIG} 8
```

To validate a model with 8 GPUs:

```
bash tools/dist_test.sh ${CONFIG} ${CEHCKPOINT} 8 --eval
```

To test a model's inference speed:

```
python tools/benchmark.py ${CONFIG} ${CEHCKPOINT}
```
### 4. Convert Model
To convert PyTorch model to ONNX:

```
python tools/export_onnx.py ${CONFIG} ${CEHCKPOINT} --simplify
```


### 5. Results on nuScenes val split 
|      Range       | $\mathrm{AP}_{ped}$ | $\mathrm{AP}_{div}$ | $\mathrm{AP}_{bound}$ | $\mathrm{AP}$ | Epoch |
|:----------------:|:-------------------:|:-------------------:|:---------------------:|:-------------:|:-----:|
| $30\times 30\ m$ |        38.9         |        47.5         |         53.7          |     46.7      |  200  |

### 6. Qualitative results on nuScenes val split 
| Front Image | Ground Truth | Pred Result |
|-------------|--------------|-------------|
|![CAM_FRONT.jpg](resources%2Fscene-0003%2F1%2Fgt%2FCAM_FRONT.jpg)|![map.jpg](resources%2Fscene-0003%2F1%2Fgt%2Fmap.jpg)|![map.jpg](resources%2Fscene-0003%2F1%2Fpred%2Fmap.jpg)|
|![CAM_FRONT.jpg](resources%2Fscene-0003%2F14%2Fgt%2FCAM_FRONT.jpg)|![map.jpg](resources%2Fscene-0003%2F14%2Fgt%2Fmap.jpg)|![map.jpg](resources%2Fscene-0003%2F14%2Fpred%2Fmap.jpg)|
|![CAM_FRONT.jpg](resources%2Fscene-0003%2F37%2Fgt%2FCAM_FRONT.jpg)|![map.jpg](resources%2Fscene-0003%2F37%2Fgt%2Fmap.jpg)|![map.jpg](resources%2Fscene-0003%2F37%2Fpred%2Fmap.jpg)|
|![CAM_FRONT.jpg](resources%2Fscene-0016%2F2%2Fgt%2FCAM_FRONT.jpg)|![map.jpg](resources%2Fscene-0016%2F2%2Fgt%2Fmap.jpg)|![map.jpg](resources%2Fscene-0016%2F2%2Fpred%2Fmap.jpg)|
|![CAM_FRONT.jpg](resources%2Fscene-0016%2F9%2Fgt%2FCAM_FRONT.jpg)|![map.jpg](resources%2Fscene-0016%2F9%2Fgt%2Fmap.jpg)|![map.jpg](resources%2Fscene-0016%2F9%2Fpred%2Fmap.jpg)|
|![CAM_FRONT.jpg](resources%2Fscene-0016%2F15%2Fgt%2FCAM_FRONT.jpg)|![map.jpg](resources%2Fscene-0016%2F15%2Fgt%2Fmap.jpg)|![map.jpg](resources%2Fscene-0016%2F15%2Fpred%2Fmap.jpg)|
|![CAM_FRONT.jpg](resources%2Fscene-0095%2F8%2Fgt%2FCAM_FRONT.jpg)|![map.jpg](resources%2Fscene-0095%2F8%2Fgt%2Fmap.jpg)|![map.jpg](resources%2Fscene-0095%2F8%2Fpred%2Fmap.jpg)|
|![CAM_FRONT.jpg](resources%2Fscene-0105%2F8%2Fgt%2FCAM_FRONT.jpg)|![map.jpg](resources%2Fscene-0105%2F8%2Fgt%2Fmap.jpg)|![map.jpg](resources%2Fscene-0105%2F8%2Fpred%2Fmap.jpg)|
|![CAM_FRONT.jpg](resources%2Fscene-0105%2F38%2Fgt%2FCAM_FRONT.jpg)|![map.jpg](resources%2Fscene-0105%2F38%2Fgt%2Fmap.jpg)|![map.jpg](resources%2Fscene-0105%2F38%2Fpred%2Fmap.jpg)|
|![CAM_FRONT.jpg](resources%2Fscene-0110%2F29%2Fgt%2FCAM_FRONT.jpg)|![map.jpg](resources%2Fscene-0110%2F29%2Fgt%2Fmap.jpg)|![map.jpg](resources%2Fscene-0110%2F29%2Fpred%2Fmap.jpg)|
|![CAM_FRONT.jpg](resources%2Fscene-0519%2F9%2Fgt%2FCAM_FRONT.jpg)|![map.jpg](resources%2Fscene-0519%2F9%2Fgt%2Fmap.jpg)|![map.jpg](resources%2Fscene-0519%2F9%2Fpred%2Fmap.jpg)|
|![CAM_FRONT.jpg](resources%2Fscene-0914%2F8%2Fgt%2FCAM_FRONT.jpg)|![map.jpg](resources%2Fscene-0914%2F8%2Fgt%2Fmap.jpg)|![map.jpg](resources%2Fscene-0914%2F8%2Fpred%2Fmap.jpg)|
|![CAM_FRONT.jpg](resources%2Fscene-1059%2F39%2Fgt%2FCAM_FRONT.jpg)|![map.jpg](resources%2Fscene-1059%2F39%2Fgt%2Fmap.jpg)|![map.jpg](resources%2Fscene-1059%2F39%2Fpred%2Fmap.jpg)|
|![CAM_FRONT.jpg](resources%2Fscene-1064%2F10%2Fgt%2FCAM_FRONT.jpg)|![map.jpg](resources%2Fscene-1064%2F10%2Fgt%2Fmap.jpg)|![map.jpg](resources%2Fscene-1064%2F10%2Fpred%2Fmap.jpg)|

## Publications

## Patents