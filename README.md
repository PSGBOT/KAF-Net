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
```
python train.py --log_name psr_hg_512_dp \                                    train
                --data_dir dir_to_psr_dataset \
                --arch fcsgg \
                --lr 5e-4 \
                --lr_step 90,120 \
                --batch_size 4 \
                --num_epochs 140 --num_workers 0 --log_interval 10
```

## Evaluate
mkdir -p ~/Reconst/Data/PSR/Simple/train
mkdir -p ~/Reconst/Data/PSR/Simple/val
cd ~/Reconst/Data/PSR/Simple
ls -d */ | grep -vE 'train|val' | shuf > all_samples.txt
head -n 120 all_samples.txt > train_samples.txt
tail -n +121 all_samples.txt > val_samples.txt
cat train_samples.txt | xargs -I{} mv {} train/
cat val_samples.txt | xargs -I{} mv {} val/

python train.py --log_name hrnet \
                --data_dir ~/Reconst/Data/PSR/Simple \
                --arch hrnet \
                --lr 5e-4 \
                --lr_step 90,180 \
                --batch_size 4 \
                --num_epochs 90 --num_workers 0 --log_interval 10
