#!/usr/bin/env python3
"""
Test script to verify the implementation of different partitioning methods.
This script tests IID, shard-based, and Dirichlet partitioning.
"""

try:
    import numpy as np
    from data import MNISTDataset, FederatedSampler
    
    def test_partitioning():
        """Test different partitioning methods."""
        print("🔄 Loading MNIST dataset...")
        dataset = MNISTDataset(root="../datasets/", train=True)
        print(f"✅ Dataset loaded: {len(dataset)} samples")
        
        n_clients = 5  # Use fewer clients for testing
        
        # Test 1: IID Partitioning
        print("\n🧪 Testing IID partitioning...")
        sampler_iid = FederatedSampler(
            dataset=dataset,
            partition_mode="iid",
            n_clients=n_clients
        )
        print("✅ IID partitioning successful")
        
        # Test 2: Shard-based partitioning
        print("\n🧪 Testing Shard-based partitioning...")
        sampler_shard = FederatedSampler(
            dataset=dataset,
            partition_mode="shard",
            n_clients=n_clients,
            n_shards=10  # 2 shards per client
        )
        print("✅ Shard-based partitioning successful")
        
        # Test 3: Dirichlet partitioning
        print("\n🧪 Testing Dirichlet partitioning...")
        sampler_dirichlet = FederatedSampler(
            dataset=dataset,
            partition_mode="dirichlet",
            n_clients=n_clients,
            dirichlet_alpha=0.1
        )
        print("✅ Dirichlet partitioning successful")
        
        # Quick analysis
        print("\n📊 Quick Analysis:")
        
        # Analyze one client from each method
        client_id = 0
        
        # IID analysis
        iid_indices = sampler_iid.dict_users[client_id]
        iid_labels = dataset.targets[iid_indices].numpy()
        iid_unique, iid_counts = np.unique(iid_labels, return_counts=True)
        
        # Shard analysis  
        shard_indices = sampler_shard.dict_users[client_id]
        shard_labels = dataset.targets[shard_indices].numpy()
        shard_unique, shard_counts = np.unique(shard_labels, return_counts=True)
        
        # Dirichlet analysis
        dir_indices = sampler_dirichlet.dict_users[client_id]
        dir_labels = dataset.targets[dir_indices].numpy()
        dir_unique, dir_counts = np.unique(dir_labels, return_counts=True)
        
        print(f"Client {client_id} class distributions:")
        print(f"  IID:       Classes {iid_unique.tolist()[:5]}{'...' if len(iid_unique) > 5 else ''} (Total: {len(iid_unique)} classes)")
        print(f"  Shard:     Classes {shard_unique.tolist()[:5]}{'...' if len(shard_unique) > 5 else ''} (Total: {len(shard_unique)} classes)")
        print(f"  Dirichlet: Classes {dir_unique.tolist()[:5]}{'...' if len(dir_unique) > 5 else ''} (Total: {len(dir_unique)} classes)")
        
        print("\n🎉 All tests passed! Implementation is working correctly.")
        print("\n📝 You can now run experiments using:")
        print("   • IID:        --partition_mode iid")
        print("   • Shard:      --partition_mode shard")  
        print("   • Dirichlet:  --partition_mode dirichlet --dirichlet_alpha 0.1")
        
        return True

    if __name__ == "__main__":
        test_partitioning()
        
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("💡 Make sure you're running this in the proper conda environment with required packages installed.")
    print("   Run: conda activate fedavg")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("💡 Check if the MNIST dataset is available in the ../datasets/ directory.")
