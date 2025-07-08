import torch
import torch.nn as nn
import torch.multiprocessing as mp
from model import NextMedPredModel, RefinePrediction
from train import (
    train_model, evaluate_model, train_refine_module,
    check_difference, uncerntain_aware_fine_tuning,
    train_mortality, evaluate_mortality
)
from conformal_fdr import nonconformalized_score
from dataloader import (
    EHRPredDataset, ReliableEHRPredDataset,
    UncerntainEHRPredDataset, MortalityEHRPredDataset
)
from torch.utils.data import DataLoader
from mortality_rate import MortalityRate
import argparse
import os
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import numpy as np

# Specify GPUs to use
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

# Argument parser
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--epochs_re', type=int, default=30)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--lr_re', type=float, default=0.0001)
    parser.add_argument('--vital_dim', type=int, default=45)
    parser.add_argument('--num_classes', type=int, default=25)
    parser.add_argument('--med_dim', type=int, default=4)
    parser.add_argument('--demo_dim', type=int, default=4)
    parser.add_argument('--note_dim', type=int, default=768)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--alpha', type=float, default=0.05)
    parser.add_argument('--gpus', type=int, default=torch.cuda.device_count())
    parser.add_argument('--backend', type=str, default='nccl')
    parser.add_argument('--c', type=float, default=0.3, help='uncertainty threshold')
    return parser.parse_args()

# Initialize distributed training
def setup(rank, world_size, backend):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29505'
    dist.init_process_group(backend, init_method='env://', world_size=world_size, rank=rank)
    torch.cuda.set_device(rank)

# Main function for each process
def main(rank, world_size):
    args = parse_args()
    setup(rank, world_size, args.backend)

    print('Loading the dataset...')
    dataset = EHRPredDataset('sepsis_time_series_data.csv')
    train_size = int(0.5 * len(dataset))
    val_size = int(0.45 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

    # Distributed samplers
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    # Data loaders
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, sampler=test_sampler)

    # Datasets for refinement and uncertainty modules
    reliable_dataset = ReliableEHRPredDataset('sepsis_time_series_data.csv')
    train_dataset_re, val_dataset_re, test_dataset_re = torch.utils.data.random_split(
        reliable_dataset, [int(0.8 * len(reliable_dataset)), int(0.1 * len(reliable_dataset)), len(reliable_dataset) - int(0.8 * len(reliable_dataset)) - int(0.1 * len(reliable_dataset))]
    )
    train_dataloader_re = DataLoader(train_dataset_re, batch_size=args.batch_size, sampler=DistributedSampler(train_dataset_re, world_size, rank, shuffle=True))
    val_dataloader_re = DataLoader(val_dataset_re, batch_size=args.batch_size, sampler=DistributedSampler(val_dataset_re, world_size, rank, shuffle=True))
    test_dataloader_re = DataLoader(test_dataset_re, batch_size=args.batch_size, sampler=DistributedSampler(test_dataset_re, world_size, rank, shuffle=True))

    uncerntain_dataset = UncerntainEHRPredDataset('sepsis_time_series_data.csv')
    uncertain_dataloader = DataLoader(uncerntain_dataset, batch_size=args.batch_size, shuffle=True)

    mortality_dataset = MortalityEHRPredDataset('sepsis_time_series_data.csv')
    train_dataset_m, val_dataset_m, test_dataset_m = torch.utils.data.random_split(
        mortality_dataset, [int(0.8 * len(mortality_dataset)), int(0.1 * len(mortality_dataset)), len(mortality_dataset) - int(0.9 * len(mortality_dataset))]
    )
    train_dataloader_m = DataLoader(train_dataset_m, batch_size=args.batch_size, sampler=DistributedSampler(train_dataset_m, world_size, rank, shuffle=True))
    val_dataloader_m = DataLoader(val_dataset_m, batch_size=args.batch_size, sampler=DistributedSampler(val_dataset_m, world_size, rank, shuffle=False))
    test_dataloader_m = DataLoader(test_dataset_m, batch_size=args.batch_size, sampler=DistributedSampler(test_dataset_m, world_size, rank, shuffle=False))

    # Initialize model
    print('Initializing the model...')
    model = NextMedPredModel(args.vital_dim, args.num_classes, args.med_dim, args.note_dim, args.d_model, args.n_heads, args.n_layers).to(rank)
    refine_module = RefinePrediction(args.d_model, args.num_classes).to(rank)
    mortality_model = MortalityRate(args.num_classes, args.d_model, args.vital_dim, args.note_dim, args.demo_dim).to(rank)

    # Load checkpoints if available
    model_path = f'model_weights_{rank}.pt'
    refine_module_path = f'refine_module_weights_{rank}.pt'
    mortality_model_path = f'mortality_model_weights_{rank}.pt'

    if os.path.exists(model_path) and os.path.exists(refine_module_path):
        model.load_state_dict(torch.load(model_path, map_location=f'cuda:{rank}'))
        refine_module.load_state_dict(torch.load(refine_module_path, map_location=f'cuda:{rank}'))
        print('Model weights loaded successfully!')
        max_diff, min_diff = check_difference(model, refine_module, train_dataloader_re, uncertain_dataloader, rank)
    else:
        print('No model weights found! Start training from scratch...')
        model = nn.parallel.DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=True)
        refine_module = nn.parallel.DistributedDataParallel(refine_module, device_ids=[rank], find_unused_parameters=True)

        print('Training the model...')
        train_model(model, train_dataloader, val_dataloader, args, rank)
        print('Evaluating before fine-tuning...')
        evaluate_model(model, test_dataloader_re, rank, args)
        print('Training refine module...')
        train_refine_module(model, refine_module, train_dataloader_re, val_dataloader_re, args, rank)
        print('Computing uncertainty difference...')
        max_diff, min_diff = check_difference(model, refine_module, train_dataloader_re, uncertain_dataloader, rank)
        print('Uncertainty-aware fine-tuning...')
        uncerntain_aware_fine_tuning(model, refine_module, train_dataloader, val_dataloader, args, rank, max_diff, min_diff)
        print('Evaluating after fine-tuning...')
        evaluate_model(model, test_dataloader_re, rank, args)
        torch.save(model.module.state_dict(), model_path)
        torch.save(refine_module.module.state_dict(), refine_module_path)

    # Mortality prediction module
    print('Training mortality model...')
    train_mortality(model, mortality_model, train_dataloader_m, val_dataloader_m, args, rank)
    print('Evaluating mortality model...')
    evaluate_mortality(model, mortality_model, test_dataloader_m, rank, args)

    # Conformal selection and FDR control 
    print('Conformal selection and FDR evaluation...')
    emb_calib, calibration_scores, emb_test, test_scores = nonconformalized_score(
        model, refine_module, train_dataloader, val_dataloader, test_dataloader, max_diff, min_diff, args, rank
    )

    # Convert to tensor for collective communication
    emb_calib, calibration_scores = torch.tensor(emb_calib).to(rank), torch.tensor(calibration_scores).to(rank)
    emb_test, test_scores = torch.tensor(emb_test).to(rank), torch.tensor(test_scores).to(rank)

    # Gather tensors from all processes
    gathered_emb_calib = [torch.zeros_like(emb_calib) for _ in range(world_size)]
    gathered_calibration_scores = [torch.zeros_like(calibration_scores) for _ in range(world_size)]
    gathered_emb_test = [torch.zeros_like(emb_test) for _ in range(world_size)]
    gathered_test_scores = [torch.zeros_like(test_scores) for _ in range(world_size)]
    dist.all_gather(gathered_emb_calib, emb_calib)
    dist.all_gather(gathered_calibration_scores, calibration_scores)
    dist.all_gather(gathered_emb_test, emb_test)
    dist.all_gather(gathered_test_scores, test_scores)

    if rank == 0:
        # Optionally save all gathered tensors
        gathered_emb_calib = torch.cat(gathered_emb_calib, dim=0)
        gathered_calibration_scores = torch.cat(gathered_calibration_scores, dim=0)
        gathered_emb_test = torch.cat(gathered_emb_test, dim=0)
        gathered_test_scores = torch.cat(gathered_test_scores, dim=0)

    dist.destroy_process_group()

# Multi-process launcher
if __name__ == "__main__":
    args = parse_args()
    world_size = 4  # number of GPUs
    mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)
