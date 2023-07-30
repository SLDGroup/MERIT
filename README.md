# MERIT

This is the implementation of [Multi-scale Hierarchical Vision Transformer with Cascaded Attention Decoding for Medical Image Segmentation, MIDL 2023](https://2023.midl.io/papers/p165) [Video](https://youtu.be/DYwsK2lmhm4). 

## Architectures

<p align="center">
<img src="figures/cascaded_merit_architecture.jpg" width=80% height=50% 
class="center">
</p>

<p align="center">
<img src="figures/parallel_merit_architecture.jpg" width=80% height=50% 
class="center">
</p>

## Qualitative Results on Synapse Multi-organ dataset

<p align="center">
<img src="figures/qualitative_results.png" width=80% height=50% 
class="center">
</p>

## Usage:
### Recommended environment:
```
Python 3.8
Pytorch 1.11.0
torchvision 0.12.0
```
Please use ```pip install -r requirements.txt``` to install the dependencies.

### Data preparation:
- **Synapse Multi-organ dataset:**
Sign up in the [official Synapse website](https://www.synapse.org/#!Synapse:syn3193805/wiki/89480) and download the dataset. Then split the 'RawData' folder into 'TrainSet' (18 scans) and 'TestSet' (12 scans) following the [TransUNet's](https://github.com/Beckschen/TransUNet/blob/main/datasets/README.md) lists and put in the './data/synapse/Abdomen/RawData/' folder. Finally, preprocess using ```python ./utils/preprocess_synapse_data.py``` or download the [preprocessed data](https://drive.google.com/file/d/1tGqMx-E4QZpSg2HQbVq5W3KSTHSG0hjK/view?usp=share_link) and save in the './data/synapse/' folder. 
Note: If you use the preprocessed data from [TransUNet](https://drive.google.com/drive/folders/1ACJEoTp-uqfFJ73qS3eUObQh52nGuzCd), please make necessary changes (i.e., remove the code segment (line# 88-94) to convert groundtruth labels from 14 to 9 classes) in the utils/dataset_synapse.py. 

- **ACDC dataset:**
Download the preprocessed ACDC dataset from [Google Drive of MT-UNet](https://drive.google.com/file/d/13qYHNIWTIBzwyFgScORL2RFd002vrPF2/view) and move into './data/ACDC/' folder.

### Pretrained model:
You should download the pretrained MaxViT models from [Google Drive](https://drive.google.com/drive/folders/1k-s75ZosvpRGZEWl9UEpc_mniK3nL2xq?usp=share_link), and then put it in the './pretrained_pth/maxvit/' folder for initialization.

### Training:
```
cd into MERIT
```

For Synapse Multi-organ training run ```CUDA_VISIBLE_DEVICES=0 python -W ignore train_synapse.py```

For ACDC training run ```CUDA_VISIBLE_DEVICES=0 python -W ignore train_ACDC.py```

### Testing:
```
cd into MERIT 
```

For Synapse Multi-organ testing run ```CUDA_VISIBLE_DEVICES=0 python -W ignore test_synapse.py```

For ACDC testing run ```CUDA_VISIBLE_DEVICES=0 python -W ignore test_ACDC.py```

## Acknowledgement
We are very grateful for these excellent works [timm](https://github.com/huggingface/pytorch-image-models), [CASCADE](https://github.com/SLDGroup/CASCADE), [PraNet](https://github.com/DengPingFan/PraNet), [Polyp-PVT](https://github.com/DengPingFan/Polyp-PVT) and [TransUNet](https://github.com/Beckschen/TransUNet), which have provided the basis for our framework.

## Citations

``` 
@inproceedings{rahman2023multi,
  title={Multi-scale Hierarchical Vision Transformer with Cascaded Attention Decoding for Medical Image Segmentation},
  author={Rahman, Md Mostafijur and Marculescu, Radu},
  booktitle={Medical Imaging with Deep Learning (MIDL)},
  month={July},
  year={2023}
}
```
