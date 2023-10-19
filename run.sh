#!/bin/bash

#python inference.py --output_dir debug --checkpoint /home/jano1906/git/deit/checkpoints/compdeitsmall_ep630.pth --evaluate evaluate_01_816
python inference.py --output_dir debug --count_flops
#python main.py --sample_divisions --finetune checkpoints/compdeitsmall_ep630.pth --output_dir trash --model deit_small_patch16_LS --data-set CIFAR10 --data-path /home/jano1906/datasets/cifar10 --batch 64 --lr 1e-4 --epochs 10 --weight-decay 0.05 --sched cosine --input-size 224 --eval-crop-ratio 1.0 --reprob 0.0 --smoothing 0.0 --warmup-epochs 5 --drop 0.0 --seed 0 --opt adamw --warmup-lr 1e-6 --mixup .8 --drop-path 0.05 --cutmix 1.0 --unscale-lr --repeated-aug --bce-loss  --color-jitter 0.3 --ThreeAugment

