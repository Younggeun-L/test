## Overview
This files are for making Single RGB image-based foreground/background 2.5D data composition and image post-processing testbed 
<img scr = "image1.png">

## Datasets
FFHQ (512 X 512) dataset can be downloaded from [Kaggle] (https://www.kaggle.com/arnaud58/flickrfaceshq-dataset-ffhq)
COCO dataset can be downloaded from [COCOdataset] (https://cocodataset.org/#download)
Use Our Mask files from mask folder or make your own mask files
Prepare your own dataset to use your dataset 

## Requirements and install
1. Create virtual environment
Create virtual environment with python version 3.7. Then Activate your virtual environment
```bash
conda create –n test python=3.7
conda activate test
```

2. Install requirements
```bash
pip install -r requirements.txt
```
 or
```bash
pip install numpy
pip install pillow
pip install scikit-learn
pip install tqdm
pip install matplotlib
pip install visdom
pip install timm
pip install opencv   or   conda install -c conda-forge opencv
```

3. Download Pre-train model

 Download Deeplab V3+ Pre-train model from [Deeplab V3+](https://github.com/VainF/DeepLabV3Plus-Pytorch) and put it in DeepLabV3/checkpoints folder

 Download MiDaS Pre-train model from [MiDaS](https://github.com/isl-org/MiDaS) and put it in MiDaS/weights folder

 Download Co-Modulated-GAN Pre-trian model using the command below and put it in checkpoints folder
```bash
download places512.sh   or    download ffhq512.sh
```

## RUN

1. Run the command below to test our file

```bash
python test.py -i [input folder]
```


2. Run the command below to test about FFHQ 

(This does not make  segment, fore/backgournd, mask and depth images) 

```bash
python test_ffhq.py -i [input folder(ffhq images)] -m [mask folder]
```

3. To test one image from uploading in local, use test_one_image.ipynb
first install jupyter_innotater
```bash
pip install jupyter_innotater
```
Then run cells in test_one_image.ipynb at jupyter lab

## Evalutate

If you want to evaluate ssim score from input and output, use the command below 

```bash
python ssim_eval.py -f [first(input) image folder] -s [second(output) image folder]
```

## Reference
```bash
@inproceedings{zhao2021comodgan,
  title={Large Scale Image Completion via Co-Modulated Generative Adversarial Networks},
  author={Zhao, Shengyu and Cui, Jonathan and Sheng, Yilun and Dong, Yue and Liang, Xiao and Chang, Eric I and Xu, Yan},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2021}
```
```bash
@article{Ranftl2020,
	author    = {Ren\'{e} Ranftl and Katrin Lasinger and David Hafner and Konrad Schindler and Vladlen Koltun},
	title     = {Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-shot Cross-dataset Transfer},
	journal   = {IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)},
	year      = {2020},
}
```
```bash
@inproceedings{deeplabv3plus2018,
  title={Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation},
  author={Liang-Chieh Chen and Yukun Zhu and George Papandreou and Florian Schroff and Hartwig Adam},
  booktitle={ECCV},
  year={2018}
}
```
