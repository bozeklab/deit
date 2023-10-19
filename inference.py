import torch
import argparse
import json
import os
from timm.models import create_model
from pathlib import Path
from tqdm import tqdm
from datasets import build_dataset
import utils
from timm.utils import accuracy
import numpy as np
import models
import models_v2
from mask_const import sample_masks, get_division_masks_for_model, DIVISION_IDS, DIVISION_MASKS
from functools import partial
from fvcore.nn import FlopCountAnalysis
import pandas as pd

def get_args_parser():
    parser = argparse.ArgumentParser('DeiT training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=64, type=int)
    # Model parameters
    parser.add_argument('--model', default='deit_small_patch16_LS', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input_size', default=224, type=int, help='images input size')
    parser.add_argument('--eval-crop-ratio', default=1., type=float, help="Crop ratio for evaluation")

    parser.add_argument('--data-path', default=f'~/datasets/imagenet/ILSVRC/Data/CLS-LOC/', type=str,
                        help='dataset path')
    parser.add_argument('--data-set', default='IMNET', choices=['CIFAR', 'IMNET', 'INAT', 'INAT19'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')

    parser.add_argument('--output_dir', type=str, required=True,
                        help='path where to save')
    parser.add_argument('--checkpoint', default=None, help="path to model checkpoint")
    parser.add_argument('--device', default="cuda", type=str)
    
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--distributed', action='store_true', default=False, help='Enabling distributed training')
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    parser.add_argument('--evaluate', nargs='*', default=[])
    parser.add_argument('--extract', nargs='*', default=[])
    parser.add_argument('--count_flops', action="store_true")
    
    parser.add_argument('--group_by_class', action="store_true")
    
    return parser



@torch.no_grad()
def evaluate(data_loader, model, device, KMs, seq: bool=False, group_by_class: bool=False):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    
    # switch to evaluation mode
    model.eval()
    division_masks = get_division_masks_for_model(model)

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        if group_by_class:
            classes = set(target.cpu().numpy())
            grouped = {c: [images[target == c].clone(), torch.full(((target == c).sum().item(),), c, device=target.device)] for c in classes}
        else:
            classes = [0]
            grouped = {0: [images, target]}
        for c, [images, target] in grouped.items():
            # compute output
            with torch.cuda.amp.autocast():
                outputs = [
                    [[k, m], model(images, K=k, masks=sample_masks(division_masks, m))]
                    for k, m in KMs
                ]
            accuracies = [
                [[k, m, i], accuracy(out, target)[0]]
                for [[k, m], outs] in outputs
                for i, out in (enumerate(outs) if seq else [[0, outs]])
            ]

            batch_size = images.shape[0]
            for [[k, m, i], acc] in accuracies:
                name = ''
                if group_by_class:
                    name += f'cls{c}_'
                name += 'acc1'
                if [k, m] != [0, 1]:
                    name += f"_K{k}_M{m}"
                    if seq:
                        name += f"_i{i}"
                metric_logger.meters[name].update(acc.item(), n=batch_size)
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



@torch.no_grad()
def extract(data_loader, model, device, KMs, seq: bool=False, group_by_class: bool=False):
    # switch to evaluation mode
    model.eval()
    division_masks = get_division_masks_for_model(model)
    ret = {}
    for images, target in tqdm(data_loader, "extracting features "):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        if group_by_class:
            classes = set(target.cpu().numpy())
            grouped = {c: [images[target == c].clone(), torch.full(((target == c).sum().item(),), c, device=target.device)] for c in classes}
        else:
            classes = [0]
            grouped = {0: [images, target]}
        for c, [images, target] in grouped.items():
            # compute output
            with torch.cuda.amp.autocast():
                outputs = [
                    [[k, m], model(images, K=k, masks=sample_masks(division_masks, m), seq=seq, cls_only=True).cpu().numpy()]
                    for k, m in KMs
                ]
            for [[k, m], feat] in outputs:
                name = ''
                if group_by_class:
                    name += f'cls{c}_'
                name += 'feat'
                if [k, m] != [0, 1]:
                    name += f"_K{k}_M{m}"
                if name not in ret:
                    ret[name] = feat
                else:
                    ret[name] = np.concatenate([ret[name], feat], axis=0)
    return ret

def evaluate_km(data_loader, model, device, group_by_class, *args, **kwargs):
    KMs = [[k, m] for m in model.division_masks.keys() for k in range(len(model.blocks))]
    return evaluate(data_loader, model, device, KMs=KMs, group_by_class=group_by_class)

def evaluate_01_816(data_loader, model, device, group_by_class, *args, **kwargs):
    KMs = [[0,1], [8,16]]
    return evaluate(data_loader, model, device, KMs=KMs, group_by_class=group_by_class)

def evaluate_seq(data_loader, model, device, group_by_class, *args, **kwargs):
    KMs = [[k, max(model.division_masks.keys())] for k in range(len(model.blocks))]
    return evaluate(data_loader, model, device, KMs=KMs, seq=True, group_by_class=group_by_class)

def count_flops(create_model_fn, img_size):
    IMG = torch.zeros(1,3,img_size,img_size)
    division_masks = DIVISION_MASKS[14][16][0]
    division_ids = DIVISION_IDS[14][16][0]
    imgs = []
    for divm in division_masks:
        divm = np.expand_dims(divm, [0,1]).repeat(3, axis=1).repeat(16, axis=2).repeat(16, axis=3)
        H, W = divm.sum(axis=2).max(), divm.sum(axis=3).max()
        imgs.append(IMG[divm].reshape(1,3,H, W))

    with torch.no_grad():
        flops = {}
        for k in tqdm(range(len(create_model_fn().blocks)), f"K: "):
            flops[k] = []
            model = create_model_fn()
            model.comp_next_init()
            cache = model._comp_next_cache
            for i, [img, id] in enumerate(zip(imgs, division_ids)):
                model = create_model_fn()
                model.forward = partial(model.comp_next, K=k, ids=id)
                model.eval()
                model._comp_next_cache = cache
                flops[k].append(FlopCountAnalysis(model, img).total())
                cache = model._comp_next_cache
                assert len(cache["xs_feats"]) == i+1, len(cache["xs_feats"])
                assert cache["i"] == i+1, cache["i"]
    return flops


def extract_k(data_loader, model, device, group_by_class, *args, **kwargs):
    KMs = [[k, max(model.division_masks.keys())] for k in range(len(model.blocks))]
    return extract(data_loader, model, device, KMs=KMs, seq=False, group_by_class=group_by_class)

def extract_01(data_loader, model, device, group_by_class, *args, **kwargs):
    KMs = [[0,1]]
    return extract(data_loader, model, device, KMs=KMs, seq=False, group_by_class=group_by_class)



def main(args):
    output_dir = Path(args.output_dir)
    with open(os.path.join(output_dir, "args.json"), "w") as f:
        json.dump(args.__dict__, f, indent=2)

    utils.init_distributed_mode(args)
    dataset_val, nb_classes = build_dataset(is_train=False, args=args)
    with open(os.path.join(output_dir, "class_to_idx.json"), "a") as f:
        f.write(json.dumps(dataset_val.class_to_idx))
    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        if len(dataset_val) % num_tasks != 0:
            print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                  'This will slightly alter validation results as extra duplicate entries are added to achieve '
                  'equal num of samples per-process.')
        sampler_val = torch.utils.data.DistributedSampler(
            dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    model = create_model(
        args.model,
        pretrained=False,
        num_classes=nb_classes,
        img_size=args.input_size
    )
    model = model.to(args.device)
    model_without_ddp = model

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])

        for task_name in args.evaluate:
            task = globals()[task_name]
            test_stats = task(data_loader_val, model, args.device, args.group_by_class)
            with open(os.path.join(output_dir, task_name + ".txt"), "w") as f:
                f.write(json.dumps(test_stats))

        for task_name in args.extract:
            assert not args.distributed
            task = globals()[task_name]
            test_stats = task(data_loader_val, model, args.device, args.group_by_class)
            np.savez_compressed(os.path.join(output_dir, task_name), **test_stats)
    else:
        print("[WARNING] --checkpoint is None")
    
    if args.count_flops:
        create_model_fn = partial(create_model,
            args.model,
            pretrained=False,
            num_classes=nb_classes,
            img_size=args.input_size
        )
        flops = count_flops(create_model_fn, args.input_size)
        flat_flops = [(str(k),i, v/1000000000) for k, l in flops.items() for i, v in enumerate(l)]
        df = pd.DataFrame(flat_flops, columns=["K", "i", "GFLOPs"])
        pd.DataFrame.to_csv(df, os.path.join(output_dir, "count_flops.csv"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=args.output_dir == "debug")
    main(args)
