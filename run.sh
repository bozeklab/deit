#!/bin/bash

#python run_with_submitit.py --memfs_imnet --log_to_wandb --output_dir trash --model deit_small_patch16_LS --data-path /net/tscratch/people/plgapardyl/imagenet --batch 256 --lr 4e-3 --epochs 1 --weight-decay 0.05 --sched cosine --input-size 224 --eval-crop-ratio 1.0 --reprob 0.0 --nodes 1 --ngpus 8 --smoothing 0.0 --warmup-epochs 5 --drop 0.0 --seed 0 --opt lamb --warmup-lr 1e-6 --mixup .8 --drop-path 0.05 --cutmix 1.0 --unscale-lr --repeated-aug --bce-loss --color-jitter 0.3 --ThreeAugment
#python main.py --memfs_imnet --log_to_wandb --output_dir trash --model deit_small_patch16_LS --data-path /net/tscratch/people/plgapardyl/imagenet --batch 64 --lr 4e-3 --epochs 1 --weight-decay 0.05 --sched cosine --input-size 224 --eval-crop-ratio 1.0 --reprob 0.0 --smoothing 0.0 --warmup-epochs 5 --drop 0.0 --seed 0 --opt lamb --warmup-lr 1e-6 --mixup .8 --drop-path 0.05 --cutmix 1.0 --unscale-lr --repeated-aug --bce-loss  --color-jitter 0.3 --ThreeAugment
#python main.py --log_to_wandb --output_dir trash --model deit_small_patch16_LS --data-path /home/jan.olszewski/datasets/inet10 --batch 64 --lr 4e-3 --epochs 10 --weight-decay 0.05 --sched cosine --input-size 224 --eval-crop-ratio 1.0 --reprob 0.0 --smoothing 0.0 --warmup-epochs 5 --drop 0.0 --seed 0 --opt lamb --warmup-lr 1e-6 --mixup .8 --drop-path 0.05 --cutmix 1.0 --unscale-lr --repeated-aug --bce-loss  --color-jitter 0.3 --ThreeAugment


srun --account=plgiris-gpu-a100 --partition=plgrid-gpu-a100 --gpus-per-node=1 --nodes=1 --ntasks-per-node=1 --time=01:00:00 --pty bash -i
