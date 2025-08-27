from typing import Dict, List, Optional, Sequence
import numpy as np
import torch
from torch.utils.data import Sampler


class FederatedSampler(Sampler):
    def __init__(
        self,
        dataset: Sequence,
        partition_mode: str = "iid",
        n_clients: Optional[int] = 100,
        n_shards: Optional[int] = 200,
        dirichlet_alpha: Optional[float] = 0.1,
        non_iid: Optional[int] = None,  # Backward compatibility
    ):
        """Sampler for federated learning supporting IID, shard-based, and Dirichlet partitioning.

        Args:
            dataset (Sequence): Dataset to sample from.
            partition_mode (str): 'iid', 'shard', or 'dirichlet'. Defaults to 'iid'.
            n_clients (Optional[int]): Number of clients. Defaults to 100.
            n_shards (Optional[int]): Number of shards for shard-based partitioning. Defaults to 200.
            dirichlet_alpha (Optional[float]): Alpha parameter for Dirichlet distribution. Defaults to 0.1.
            non_iid (Optional[int]): Deprecated. Use partition_mode instead.
        """
        self.dataset = dataset
        self.n_clients = n_clients
        self.n_shards = n_shards
        self.dirichlet_alpha = dirichlet_alpha
        
        # Handle backward compatibility
        if non_iid is not None:
            self.partition_mode = "shard" if non_iid == 1 else "iid"
        else:
            self.partition_mode = partition_mode

        # Select partitioning method
        if self.partition_mode == "iid":
            self.dict_users = self._sample_iid()
        elif self.partition_mode == "shard":
            self.dict_users = self._sample_shard_based()
        elif self.partition_mode == "dirichlet":
            self.dict_users = self._sample_dirichlet()
        else:
            raise ValueError(f"Unknown partition_mode: {self.partition_mode}. Choose 'iid', 'shard', or 'dirichlet'.")

    def _sample_iid(self) -> Dict[int, List[int]]:
        num_items = int(len(self.dataset) / self.n_clients)
        dict_users, all_idxs = {}, [i for i in range(len(self.dataset))]

        for i in range(self.n_clients):
            dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
            all_idxs = list(set(all_idxs) - dict_users[i])

        return dict_users

    def _sample_shard_based(self) -> Dict[int, List[int]]:
        num_imgs = len(self.dataset) // self.n_shards  # 300

        idx_shard = [i for i in range(self.n_shards)]
        dict_users = {i: np.array([]) for i in range(self.n_clients)}
        idxs = np.arange(self.n_shards * num_imgs)
        labels = self.dataset.train_labels.numpy()

        # sort labels
        idxs_labels = np.vstack((idxs, labels))
        idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
        idxs = idxs_labels[0, :]

        # divide and assign 2 shards/client
        for i in range(self.n_clients):
            rand_set = set(np.random.choice(idx_shard, 2, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand * num_imgs : (rand + 1) * num_imgs]),
                    axis=0,
                )

        return dict_users

    def _sample_dirichlet(self) -> Dict[int, List[int]]:
        """Sample data using Dirichlet distribution for heterogeneous non-IID setting."""
        # Get labels from dataset
        if hasattr(self.dataset, 'train_labels'):
            labels = self.dataset.train_labels.numpy()
        elif hasattr(self.dataset, 'targets'):
            labels = self.dataset.targets.numpy()
        else:
            raise AttributeError("Dataset must have 'train_labels' or 'targets' attribute")
        
        n_classes = len(np.unique(labels))
        dict_users = {i: [] for i in range(self.n_clients)}
        
        # For each class, distribute its samples across clients using Dirichlet
        for class_id in range(n_classes):
            # Get indices for this class
            class_indices = np.where(labels == class_id)[0]
            np.random.shuffle(class_indices)
            
            # Sample proportions from Dirichlet distribution
            proportions = np.random.dirichlet([self.dirichlet_alpha] * self.n_clients)
            
            # Calculate split points based on proportions
            split_points = (np.cumsum(proportions) * len(class_indices)).astype(int)[:-1]
            
            # Split indices and assign to clients
            client_splits = np.split(class_indices, split_points)
            for client_id, client_indices in enumerate(client_splits):
                dict_users[client_id].extend(client_indices.tolist())
        
        # Shuffle each client's data to ensure randomness within client
        for client_id in range(self.n_clients):
            np.random.shuffle(dict_users[client_id])
            
        return dict_users

    def set_client(self, client_id: int):
        self.client_id = client_id

    def __iter__(self):
        # fetch dataset indexes based on current client
        client_idxs = list(self.dict_users[self.client_id])
        for item in client_idxs:
            yield int(item)
