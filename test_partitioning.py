#!/usr/bin/env python3
"""
Test script to demonstrate different data partitioning methods.
This script shows how to use IID, Shard-based, and Dirichlet-based partitioning.
"""

import numpy as np
from data import MNISTDataset, FederatedSampler


def analyze_data_distribution(dict_users, dataset, partition_mode, n_clients=5):
    """Analyze and print data distribution statistics for the first few clients."""
    print(f"\n=== {partition_mode.upper()} Data Distribution Analysis ===")
    
    if hasattr(dataset, 'train_labels'):
        labels = dataset.train_labels.numpy()
    elif hasattr(dataset, 'targets'):
        labels = dataset.targets.numpy()
    else:
        print("Cannot analyze: Dataset has no labels attribute")
        return
    
    for client_id in range(min(n_clients, len(dict_users))):
        client_indices = dict_users[client_id]
        if len(client_indices) == 0:
            continue
            
        client_labels = labels[client_indices]
        unique_classes, counts = np.unique(client_labels, return_counts=True)
        
        print(f"Client {client_id}: {len(client_indices)} samples")
        print(f"  Classes: {unique_classes.tolist()}")
        print(f"  Counts: {counts.tolist()}")
        print(f"  Class distribution: {dict(zip(unique_classes, counts))}")
        print()


def main():
    # Load dataset
    dataset = MNISTDataset(root="../datasets/", train=True)
    print(f"Total dataset size: {len(dataset)}")
    
    n_clients = 10
    
    # Test IID partitioning
    print("\n" + "="*50)
    print("Testing IID Partitioning")
    print("="*50)
    sampler_iid = FederatedSampler(
        dataset=dataset,
        partition_mode="iid",
        n_clients=n_clients
    )
    analyze_data_distribution(sampler_iid.dict_users, dataset, "iid", n_clients=5)
    
    # Test Shard-based partitioning
    print("\n" + "="*50)
    print("Testing Shard-based Partitioning")
    print("="*50)
    sampler_shard = FederatedSampler(
        dataset=dataset,
        partition_mode="shard",
        n_clients=n_clients,
        n_shards=20  # 2 shards per client
    )
    analyze_data_distribution(sampler_shard.dict_users, dataset, "shard", n_clients=5)
    
    # Test Dirichlet partitioning with different alpha values
    for alpha in [0.1, 0.5, 1.0]:
        print("\n" + "="*50)
        print(f"Testing Dirichlet Partitioning (alpha={alpha})")
        print("="*50)
        sampler_dirichlet = FederatedSampler(
            dataset=dataset,
            partition_mode="dirichlet",
            n_clients=n_clients,
            dirichlet_alpha=alpha
        )
        analyze_data_distribution(sampler_dirichlet.dict_users, dataset, f"dirichlet (α={alpha})", n_clients=5)


if __name__ == "__main__":
    main()
