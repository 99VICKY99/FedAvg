from typing import Any, Dict, List
import argparse
import os
import copy
import torch
import wandb


class Logger:
    def __init__(self, args):
        self.args = args
        if args.wandb:
            wandb.init(project=args.wandb_project, name=args.exp_name, config=args)
            self.wandb = wandb
        else:
            self.wandb = None

    def log(self, logs: Dict[str, Any]) -> None:
        if self.wandb:
            self.wandb.log(logs)


def average_weights(weights: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    weights_avg = copy.deepcopy(weights[0])

    for key in weights_avg.keys():
        for i in range(1, len(weights)):
            weights_avg[key] += weights[i][key]
        weights_avg[key] = torch.div(weights_avg[key], len(weights))

    return weights_avg


def arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, default="../datasets/")
    parser.add_argument("--model_name", type=str, default="cnn")

    # Data partitioning arguments
    parser.add_argument("--partition_mode", type=str, default="iid", 
                       choices=["iid", "shard", "dirichlet"],
                       help="Data partitioning method: 'iid', 'shard', or 'dirichlet'")
    parser.add_argument("--dirichlet_alpha", type=float, default=0.1,
                       help="Alpha parameter for Dirichlet distribution (lower = more non-IID)")
    parser.add_argument("--non_iid", type=int, default=1)  # Kept for backward compatibility
    parser.add_argument("--n_clients", type=int, default=100)
    parser.add_argument("--n_shards", type=int, default=200)
    parser.add_argument("--frac", type=float, default=0.1)

    parser.add_argument("--n_epochs", type=int, default=1000)
    parser.add_argument("--n_client_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--optim", type=str, default="sgd")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--early_stopping", type=int, default=1)

    parser.add_argument("--device", type=int, default=0)

    parser.add_argument("--wandb", action='store_true', help="Enable wandb logging")
    parser.add_argument("--wandb_project", type=str, default="FedAvg")
    parser.add_argument("--exp_name", type=str, default="exp")

    return parser.parse_args()
