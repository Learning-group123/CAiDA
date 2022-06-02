# PyTorch Implemetation for CAiDA

## [[NeurIPS-2021] Confident Anchor-Induced Multi-Source Free Domain Adaptation](https://proceedings.neurips.cc/paper/2021/hash/168908dd3227b8358eababa07fcaf091-Abstract.html)

This is the implementation code of our paper "**Confident Anchor-Induced Multi-Source Free Domain Adaptation**" accepted by [NeurIPS-2021](https://nips.cc/Conferences/2021). 

## Overview of The Proposed Model
![overview](./fig/overview.png)


## Requirements:

* python == 3.6.8
* pytorch == 1.1.0
* numpy == 1.17.4
* torchvision == 0.3.0
* scipy == 1.3.1
* sklearn == 0.5.0
* argparse, PIL

## Datasets Preparation:
* **Office Dataset:** Download the datasets [Office-31](https://drive.google.com/file/d/0B4IapRTv9pJ1WGZVd1VDMmhwdlE/view?resourcekey=0-gNMHVtZfRAyO_t2_WrOunA), [Office-Home](https://drive.google.com/file/d/0B81rNlvomiwed0V1YUxQdC1uOTg/view?resourcekey=0-2SNWq0CDAuWOBRRBL7ZZsw), [Office-Caltech](http://www.vision.caltech.edu/Image_Datasets/Caltech256/256_ObjectCategories.tar) from the official websites.
* **Digits-Five Dataset:** Download the datasets [MNIST](http://yann.lecun.com/exdb/mnist/), [MNIST-M](https://github.com/VanushVaswani/keras_mnistm/releases/download/1.0/keras_mnistm.pkl.gz), [USPS](https://www.kaggle.com/datasets/bistaumanga/usps-dataset), [SVHN](http://ufldl.stanford.edu/housenumbers/), [Synthetic Digits](https://www.kaggle.com/datasets/prasunroy/synthetic-digits) from the official websites.
* **DomainNet Dataset:** Download [DomainNet](http://ai.bu.edu/DomainNet/) from the official website.
* Place these datasets in './datasets'.
* Using gen_list.py to generate '.txt' file for each dataset (change dataset argument in the file accordingly).

## Training:

* Train source models (shown here for Office with source A)

```shell
python train_source.py --dset office-31 --s 0 --max_epoch 100 --trte val --gpu_id 0 --output ckps/source/
```

* Adapt to target (shown here for Office with target D)

```shell
python train_source.py --dset office-31 --t 1 --max_epoch 15 --gpu_id 0 --cls_par 0.7 --crc_par 0.01 --output_src ckps/source/ --output ckps/CAiDA
```

## Citation
If you find this code is useful to your research, please consider to cite our paper.

```
@inproceedings{NEURIPS2021_168908dd,
 author = {Dong, Jiahua and Fang, Zhen and Liu, Anjin and Sun, Gan and Liu, Tongliang},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {M. Ranzato and A. Beygelzimer and Y. Dauphin and P.S. Liang and J. Wortman Vaughan},
 pages = {2848--2860},
 publisher = {Curran Associates, Inc.},
 title = {Confident Anchor-Induced Multi-Source Free Domain Adaptation},
 volume = {34},
 year = {2021}
}
@ARTICLE{9616392,
  author={Dong, Jiahua and Cong, Yang and Sun, Gan and Fang, Zhen and Ding, Zhengming},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={Where and How to Transfer: Knowledge Aggregation-Induced Transferability Perception for Unsupervised Domain Adaptation}, 
  year={2021},
}
```

## Contact:
* **Jiahua Dong:** dongjiahua1995@gmail.com
* **Zhen Fang:**  fzjlyt@gmail.com
