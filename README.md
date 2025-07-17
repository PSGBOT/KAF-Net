# Pytorch Implementation of KAF-Net

This repository is a simple pytorch implementation of KAF-Net, some of the code is taken from the [official implementation](https://github.com/xingyizhou/CenterNet).

## Requirements:
- python>=3.10
- pytorch==1.13.1
- tensorboardX(optional)

## Getting Started
1. Disable cudnn batch normalization.
Open `torch/nn/functional.py` and find the line with `torch.batch_norm` and replace the `torch.backends.cudnn.enabled` with `False`.

2. Clone this repo:
    ```
    KAFNet_ROOT=/path/to/clone/KAF-Net
    git clone https://github.com/PSGBot/KAF-Net $KAFNet_ROOT
    ```


3. Compile deformable convolutional (from [DCNv2](https://github.com/Chen-Yulin/DCNv2)).
    ```
    cd $KAFNet_ROOT/lib/DCNv2
    ./make.sh
    ```

4. Compile NMS.
    ```
    cd $KAFNet_ROOT/lib/nms
    make
    ```

5. For PSR training, Download PSR dataset and put samples (or create symlinks) into ```$KAFNet_ROOT/data/PSR```


6. To train Hourglass-104, download [CornerNet pretrained weights (password: y1z4)](https://pan.baidu.com/s/1tp9-5CAGwsX3VUSdV276Fg) and put ```checkpoint.t7``` into ```$KAFNet_ROOT/ckpt/pretrain```.


## Train
### PSR Dataset
#### single GPU or multi GPU using nn.DataParallel
```bash
python train.py --log_name resdcn_all \
                --data_dir ~/cyl/Data/PSR_final/ \
                --arch resdcn_101 \
                --lr 5e-5 \
                --lr_step 40,70 \
                --batch_size 16 \
                --num_epochs 90 --num_workers 0 --log_interval 5
```

## Inference
```bash
python infer.py --image_path ./data/Sample\ PSR/Sample\ 1 --model_weights ./ckpt/dcn50_simple/checkpoint.t7 --visualize_output --visualization_dir ./debug_viz/infer
```
