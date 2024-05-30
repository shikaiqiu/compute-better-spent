# Scaling MLPs :fire:
​
![](https://user-images.githubusercontent.com/38691167/274573298-0aaf4b37-e9b3-4d67-9c0f-ef27fe17089d.png)
## Overview
This repository contains the code accompanying our [paper](https://arxiv.org/abs/2306.13575)  *Scaling MLPs: A Tale of Inductive Bias*. In this work we explore the limits of the *multi-layer perceptron*, or short MLP, when subjected to higher amounts of compute. More precisely, we study architectures with the following block form:
![](https://lh3.googleusercontent.com/pw/AIL4fc_3gvNmHfrvhN38zgU2OMTHqG-4w0zMY6of3S7Gi0EoV498btfYB2H7NnYUlpm8d0Va7COQAigFYZ9BCEI93qIqkV4_CKLKtdED6VQ8p-uJrKb6zD0yRfoe2yaMRdFFZeyPXaiFGWkJEurH-wvNGMY1=w1426-h154-s-no?authuser=0)
​
**Why?** We argue that such an MLP has minimal inductive bias (compared to convolutional networks, vision transformers, MLPMixers etc.) and thus offers an interesting test bed to explore whether simply scaling compute can make even the simplest models work (to some degree). The importance of inductive bias has recently been questioned due to vision transformers and MLPMixers eclipsing the more structured convolutional models on standard benchmarks.
​
Moreover, MLPs still remain to be the main protagonists in ML theory works but surprisingly, very little is known about their empirical performance at scale! We aim to close this gap here and provide the community with very performant MLPs to analyse!
​
## Explore
You can easily explore our pre-trained and fine-tuned models by specifying the pretrained flag. For instance, to load a BottleneckMLP with 12 blocks of width 1024, pre-trained on Imagenet21k, simply run
>`from models.networks import B_12_Wi_1024`
>` `
>`model = B_12_Wi_1024(dim_in=64 * 64 * 3, dim_out=11230, pretrained=True)`

If you need an already fine-tuned model, you can specify
>`from models.networks import B_12_Wi_1024`
>` `
>`model = B_12_Wi_1024(dim_in=64 * 64 * 3, dim_out=10, pretrained='cifar10')`

Check-out the Juypter notebook *explore.ipynb* to play around with the models.
## Pretrained Models

We further publish our models pre-trained on ImageNet21k for various number of epochs at an image resolution of $64\times 64$ [here](https://drive.google.com/drive/folders/17pbKnQgftxkGW5zZGuUvN1C---DesqOW?usp=sharing). Fine-tuning the $800$ epochs models for $100$ epochs should give you roughly the following down-stream performances (check *Fine-tuning* section for hyper-parameter details)

|                  | #Params | CIFAR10 | CIFAR100 | STL10 | TinyImageNet | ImageNet | ImageNetReal
| ---------------- | ------- | ------- | -------- | ----- | ------------ | ---------- | ------------
| **B_6-Wi_512**   | 24M     | 88.5%   | 71.2%    | 79.9% |    53.2%     |    33.3%   |    38.2
| **B_12-Wi_512**  | 37M     | 91.4%   | 75.1%    | 84.4% |    60.0%     |    38.0%   |    42.8
| **B_6-Wi_1024**  | 74M     | 92.5%   | 77.1%    | 86.5% |    64.3%     |    40.0%   |    47.0%
| **B_12-Wi_1024** | 124M    | 94.2%   | 80.0%    | 89.9% |    69.9%     |    43.2%   |    48.6%
| **B_12-Wi_1024 + TTA** | 124M    | 95.5%   | 82.6%    | 92.2% |    73.1%      |     51.4%   | 57.9%


Make sure that you also download the config.txt file and place in together in the same folder as the corresponding checkpoint.
## Environment Setup
​
For installing the *FFCV* dataloading framework, we refer to the original [repository](https://github.com/libffcv/ffcv). To install the remaining packages, activate the FFCV environment and run
>`pip install -r requirements.txt`
​
## Creating .beton Files
In order to use the efficiency of MLPs to the fullest, we need a more optimised data loading framework than the standard one provided by *torch*. This is because the data transfer from CPU to GPU otherwise becomes the bottleneck of training, not the gradient computation!!
To ensure a faster data transfer, we use the *FFCV* framework, which requires converting your dataset first to the **beton** format. This can be achieved by specifying your dataset as a torchvision.dataset object.

If your dataset is implemented in the torchvision.datasets library, simply add the corresponding lines of code to the `get_dataset` function in `dataset_to_beton.py`. We provide implementations for *CIFAR10* and *CIFAR100*.

If you have your dataset in the standard hierarchical subfolder structure, i.e. your dataset consists of subfolders each corresponding to a separate class, you can simply specify the `dataset_path` argument in `create_beton` in order to obtain the *.beton* file.
​

Conversion to *.beton* accepts a resolution parameter `res`, specifying the resolution of the images. We recommend using `-- res 64` for very large datasets such as *ImageNet21k* in order to keep the computational requirements manageable for users with less resources.
​

Downloading and converting the trainset of CIFAR10 to the *.beton* format can for instance be achieved by running
>`python3 data_utils/dataset_to_beton.py --dataset_name cifar10 --mode train --res 32`

Converting a subfolder-structured dataset can be converted to the *.beton* format at resolution 64 by running
>`python3 data_utils/dataset_to_beton.py --data_path path/to/folders --mode train --res 64`

## Pre-training
​
**ImageNet21k.** Due to legal reasons, we cannot provide the *ImageNet21k* in the .beton format directly. We recommend applying [here](https://www.image-net.org/download.php) to download it but in case you cannot get access, you can use the torrent [here](https://academictorrents.com/details/8ec0d8df0fbb507594557bce993920442f4f6477). Similarly for *ImageNet1k*. Once you have downloaded the dataset, we recommend pre-processing it as detailed in this [repository](https://arxiv.org/abs/2104.10972) to remove faulty images and classes with only very little examples. Then produce the *.beton* as outlined above.
​
​

**Pre-training.** For pre-training the `B_12-Wi_1024` *BottleneckMLP* on *ImageNet21k* at resolution $64 \times 64$, you can use the following command:
>`python3 train.py --dataset imagenet21 --model BottleneckMLP --architecture B_12-Wi_1024 --batch_size 16384 --resolution 64`
​

For more specific configurations, we encourage the user to check out all available flags in `train.py`. In case you run into memory issues, try to reduce the batch-size. We remark however that smaller batch sizes tend to lead worse results, check-out our paper where we highlight this effect. During training, the parameters will automatically be saved to the `checkpoints`folder.
## Fine-tuning
​
You can fine-tune our pre-trained checkpoints or your own using the script `finetune.py`.  For instance, the following command fine-tunes the model specified in the path argument on CIFAR10, provided you have converted the CIFAR10 dataset to the *.beton* format:
> `python3 finetune.py --checkpoint_path path/to/checkpoint --dataset cifar10 --data_resolution 32 --batch_size 2048 --epochs 100 --lr 0.01 --weight_decay 0.0001 --data_path /local/home/stuff/ --crop_scale 0.4 1. --crop_ratio 1. 1. --optimizer sgd --augment --mode finetune --smooth 0.3`
​

You can also train a linear layer on top by specifying the flag `--mode linear`instead.

