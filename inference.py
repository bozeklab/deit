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
    parser.add_argument('--checkpoint', required=True, help="path to model checkpoint")
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
    
    parser.add_argument('--group_by_class', action="store_true")
    
    return parser



@torch.no_grad()
def evaluate(data_loader, model, device, KMs, seq: bool=False, group_by_class: bool=False):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    
    # switch to evaluation mode
    model.eval()

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
                    [[k, m], model((images, k, m, seq))]
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
                    [[k, m], model((images, k, m, seq, True)).cpu().numpy()]
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

def evaluate_km(data_loader, model, device, group_by_class):
    KMs = [[k, m] for m in model.division_masks.keys() for k in range(len(model.blocks))]
    return evaluate(data_loader, model, device, KMs=KMs, group_by_class=group_by_class)

def evaluate_01_816(data_loader, model, device, group_by_class):
    KMs = [[0,1], [8,16]]
    return evaluate(data_loader, model, device, KMs=KMs, group_by_class=group_by_class)

def evaluate_seq(data_loader, model, device, group_by_class):
    KMs = [[k, max(model.division_masks.keys())] for k in range(len(model.blocks))]
    return evaluate(data_loader, model, device, KMs=KMs, seq=True, group_by_class=group_by_class)

def extract_k(data_loader, model, device, group_by_class):
    KMs = [[k, max(model.division_masks.keys())] for k in range(len(model.blocks))]
    return extract(data_loader, model, device, KMs=KMs, seq=False, group_by_class=group_by_class)

def extract_01(data_loader, model, device, group_by_class):
    KMs = [[0,1]]
    return extract(data_loader, model, device, KMs=KMs, seq=False, group_by_class=group_by_class)


def main(args):

    utils.init_distributed_mode(args)
    output_dir = Path(args.output_dir)
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
    
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model_without_ddp.load_state_dict(checkpoint['model'])

    for task_name in args.evaluate:
        task = globals()[task_name]
        test_stats = task(data_loader_val, model, args.device, args.group_by_class)
        with open(os.path.join(output_dir, task_name + ".txt"), "a") as f:
            f.write(json.dumps(test_stats) + "\n")
    
    for task_name in args.extract:
        assert not args.distributed
        task = globals()[task_name]
        test_stats = task(data_loader_val, model, args.device, args.group_by_class)
        np.savez_compressed(os.path.join(output_dir, task_name), **test_stats)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
