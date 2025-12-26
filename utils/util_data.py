import os
import json
from collections import defaultdict
from typing import Dict, List, Tuple, Any

import numpy as np
import torchvision.transforms as transforms
import torch
import torch.utils.data as data
from torchvision.datasets import CIFAR10, MNIST, FashionMNIST, EMNIST
import matplotlib.pyplot as plt

from utils.datasets import CustomSVHN
from utils.tinyimagenet import TinyImageNet

from utils.util_sys import create_folder_if_not_exists

from utils.logging import logger


_DATASET_SPECS: Dict[str, Dict[str, Any]] = {
    "mnist": {"root": "mnist", "image_shape": (1, 28, 28), "num_classes": 10},
    "fmnist": {"root": "fmnist", "image_shape": (1, 28, 28), "num_classes": 10},
    "cifar10": {"root": "cifar10", "image_shape": (3, 32, 32), "num_classes": 10},
    "svhn": {"root": "svhn", "image_shape": (3, 32, 32), "num_classes": 10},
    "emnist_byclass": {"root": "emnist", "image_shape": (1, 28, 28), "num_classes": 62},
    "emnist_bymerge": {"root": "emnist", "image_shape": (1, 28, 28), "num_classes": 47},
    "tinyimagenet": {"root": "tinyimagenet", "image_shape": (3, 64, 64), "num_classes": 200},
}


class RandomShardDataset(data.Dataset):
    def __init__(
        self,
        *,
        size: int,
        image_shape: Tuple[int, ...],
        num_classes: int,
        seed: int,
    ) -> None:
        self._size = int(size)
        self._image_shape = tuple(image_shape)
        self._num_classes = int(num_classes)

        rng = np.random.default_rng(seed)
        self.targets = rng.integers(0, self._num_classes, size=self._size, dtype=np.int64).tolist()
        self.classes = list(range(self._num_classes))

        gen = torch.Generator()
        gen.manual_seed(seed)
        self._data = torch.rand((self._size, *self._image_shape), generator=gen, dtype=torch.float32)

    def __getitem__(self, index: int):
        return self._data[index], int(self.targets[index])

    def __len__(self) -> int:
        return self._size


def resolve_dataset_root(dataset_name: str, dir_data: str) -> str:
    spec = _DATASET_SPECS.get(dataset_name)
    if spec is None:
        return dir_data

    preferred_root = os.path.join(dir_data, spec["root"])
    if dataset_name == "tinyimagenet":
        legacy_root = os.path.join(dir_data, "tiny-imagenet-200")
        preferred_has_data = os.path.exists(os.path.join(preferred_root, "tiny-imagenet-200"))
        legacy_has_data = os.path.exists(legacy_root)
        if legacy_has_data and not preferred_has_data:
            return dir_data

    return preferred_root


def build_fake_datasets(
    dataset_name: str,
    *,
    seed: int,
    train_size: int = 2000,
    test_size: int = 500,
):
    spec = _DATASET_SPECS.get(dataset_name)
    if spec is None:
        raise ValueError(f"Dataset {dataset_name} not supported for fake fallback")

    train_ds = RandomShardDataset(
        size=train_size,
        image_shape=spec["image_shape"],
        num_classes=spec["num_classes"],
        seed=seed,
    )
    test_ds = RandomShardDataset(
        size=test_size,
        image_shape=spec["image_shape"],
        num_classes=spec["num_classes"],
        seed=seed + 10_000,
    )
    return train_ds, test_ds


def load_data_cifar10(
    dir_data: str,
    augmentation: bool = True
) -> Tuple[CIFAR10, CIFAR10]:

    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    transform_base = [
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]

    if augmentation:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip()
        ] + transform_base)
    else:
        transform_train = transforms.Compose(transform_base)

    transform_test = transforms.Compose(transform_base)

    cifar10_train_ds = CIFAR10(dir_data, train=True, download=True, transform=transform_train)
    cifar10_test_ds = CIFAR10(dir_data, train=False, download=True, transform=transform_test)

    return cifar10_train_ds, cifar10_test_ds


def load_data_mnist(
    dir_data: str,
) -> Tuple[MNIST, MNIST]:

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    mnist_train_ds = MNIST(dir_data, train=True, download=True, transform=transform)
    mnist_test_ds = MNIST(dir_data, train=False, download=True, transform=transform)

    return mnist_train_ds, mnist_test_ds


def load_data_fmnist(
    dir_data: str,
) -> Tuple[FashionMNIST, FashionMNIST]:

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    fmnist_train_ds = FashionMNIST(
        dir_data, train=True, download=True, transform=transform
    )
    fmnist_test_ds = FashionMNIST(
        dir_data, train=False, download=True, transform=transform
    )

    return fmnist_train_ds, fmnist_test_ds


def load_data_svhn(
    dir_data: str,
) -> Tuple[CustomSVHN, CustomSVHN]:

    transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    svhn_train_ds = CustomSVHN(
        dir_data, split="train", download=True, transform=transform
    )
    svhn_test_ds = CustomSVHN(
        dir_data, split="test", download=True, transform=transform
    )

    return svhn_train_ds, svhn_test_ds


def load_data_emnist_byclass(
    dir_data: str,
) -> Tuple[EMNIST, EMNIST]:

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    emnist_train_ds = EMNIST(
        dir_data, split="byclass", train=True, download=True, transform=transform
    )
    emnist_test_ds = EMNIST(
        dir_data, split="byclass", train=False, download=True, transform=transform
    )

    return emnist_train_ds, emnist_test_ds


def load_data_emnist_bymerge(
    dir_data: str,
) -> Tuple[EMNIST, EMNIST]:

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    emnist_train_ds = EMNIST(
        dir_data, split="bymerge", train=True, download=True, transform=transform
    )
    emnist_test_ds = EMNIST(
        dir_data, split="bymerge", train=False, download=True, transform=transform
    )

    return emnist_train_ds, emnist_test_ds


def load_data_tinyimagenet(
    dir_data: str,
) -> Tuple[TinyImageNet, TinyImageNet]:
    
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(64, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    train_ds = TinyImageNet(dir_data, train=True, download=True, transform=transform_train)
    test_ds = TinyImageNet(dir_data, train=False, download=True, transform=transform_test)
    
    return train_ds, test_ds


def get_dataset(
    dataset_name: str,
    dir_data: str,
    *,
    seed: int = 0,
    use_fake_data: bool = False,
    fake_train_size: int = 2000,
    fake_test_size: int = 500,
    allow_fallback: bool = True,
):
    """Get train/test datasets for a dataset name.

    `dir_data` is treated as a root directory; each dataset is stored under its own
    subdirectory (e.g. `<dir_data>/mnist/`, `<dir_data>/cifar10/`).
    """
    if use_fake_data:
        logger.warning("Using RandomShardDataset fake data for dataset=%s", dataset_name)
        return build_fake_datasets(
            dataset_name,
            seed=seed,
            train_size=fake_train_size,
            test_size=fake_test_size,
        )

    dataset_root = resolve_dataset_root(dataset_name, dir_data)
    create_folder_if_not_exists(dataset_root)

    try:
        if dataset_name == "mnist":
            return load_data_mnist(dataset_root)
        if dataset_name == "fmnist":
            return load_data_fmnist(dataset_root)
        if dataset_name == "cifar10":
            return load_data_cifar10(dataset_root)
        if dataset_name == "svhn":
            return load_data_svhn(dataset_root)
        if dataset_name == "emnist_byclass":
            return load_data_emnist_byclass(dataset_root)
        if dataset_name == "emnist_bymerge":
            return load_data_emnist_bymerge(dataset_root)
        if dataset_name == "tinyimagenet":
            return load_data_tinyimagenet(dataset_root)
        raise ValueError(f"Dataset {dataset_name} not supported")
    except Exception as exc:
        if not allow_fallback:
            raise
        logger.warning(
            "Failed to load dataset=%s from %s (%s). Falling back to RandomShardDataset.",
            dataset_name,
            dataset_root,
            exc,
        )
        return build_fake_datasets(
            dataset_name,
            seed=seed,
            train_size=fake_train_size,
            test_size=fake_test_size,
        )


def partition_data(
    train_ds,
    test_ds,
    num_clients: int,
    partition_type: str,
    partition_beta: float,
    min_samples_per_client: int,
    max_samples_per_client: int,
    rng: Any = None,
) -> tuple:
    """Partition the dataset among clients based on the partition type."""
    if rng is None:
        rng = np.random.default_rng()

    # Check if dataset has targets attribute
    if hasattr(train_ds, 'targets'):
        labels = np.array(train_ds.targets)
    elif hasattr(train_ds, 'labels'):
        labels = np.array(train_ds.labels)
    else:
        # Handle Subset objects or other dataset types
        if isinstance(train_ds, data.Subset):
            if hasattr(train_ds.dataset, 'targets'):
                labels = np.array([train_ds.dataset.targets[i] for i in train_ds.indices])
            elif hasattr(train_ds.dataset, 'labels'):
                labels = np.array([train_ds.dataset.labels[i] for i in train_ds.indices])
            else:
                raise ValueError("Dataset format not supported - cannot find labels")
        else:
            raise ValueError("Dataset format not supported - cannot find labels")

    if partition_type == "iid":
        user_groups = partition_data_iid(train_ds, num_clients, rng)
        return train_ds, test_ds, user_groups
    elif partition_type == "noniid":
        user_groups = partition_data_non_iid(labels, num_clients, partition_beta, rng)
        return train_ds, test_ds, user_groups
    elif partition_type == "dirichlet_fixed":
        user_groups = partition_data_dirichlet_fixed(labels, num_clients, partition_beta, rng)
        return train_ds, test_ds, user_groups
    elif partition_type == "iid_quantity_skew":
        user_groups = partition_data_iid_quantity_skew(
            labels,
            num_clients,
            min_samples_per_client,
            max_samples_per_client,
            rng,
        )
        return train_ds, test_ds, user_groups
    else:
        raise ValueError(f"Partition type {partition_type} not supported")


def partition_data_iid(train_ds, num_clients, rng: Any):
    """Partition the dataset in an IID manner."""
    num_samples = len(train_ds)
    indices = np.arange(num_samples)
    rng.shuffle(indices)
    
    # Divide indices equally among clients
    chunks = np.array_split(indices, num_clients)
    user_groups = {i: chunk.tolist() for i, chunk in enumerate(chunks)}
    
    return user_groups


def partition_data_non_iid(labels, num_clients, beta, rng: Any):
    """
    Partition the dataset in a non-IID manner using Dirichlet distribution.
    
    Args:
        labels: Array of labels for the dataset
        num_clients: Number of clients
        beta: Parameter for Dirichlet distribution
        
    Returns:
        Dictionary mapping client IDs to arrays of data indices
    """
    num_classes = len(np.unique(labels))
    user_groups = {i: [] for i in range(num_clients)}
    
    # Group indices by class
    class_indices = {c: np.where(labels == c)[0] for c in range(num_classes)}
    
    # For each class, distribute data to clients according to Dirichlet distribution
    for c in range(num_classes):
        proportions = rng.dirichlet(np.repeat(beta, num_clients))
        # Normalize proportions to sum to the number of samples in this class
        proportions = proportions / proportions.sum() * len(class_indices[c])
        proportions = proportions.astype(int)
        
        # Adjust for rounding errors
        diff = len(class_indices[c]) - proportions.sum()
        proportions[-1] += diff
        
        # Distribute indices
        indices = rng.permutation(class_indices[c])
        start_idx = 0
        for i in range(num_clients):
            end_idx = start_idx + proportions[i]
            user_groups[i].extend(indices[start_idx:end_idx].tolist())
            start_idx = end_idx
            
    return user_groups


def partition_data_dirichlet_fixed(labels, num_clients, beta, rng: Any):
    """
    Partitions data with non-IID label distribution and roughly equal quantity per client.
    1. Initially partitions data using a Dirichlet distribution, creating non-IID label distributions
       but varying sample counts per client.
    2. Calculates a target sample count per client.
    3. Balances the number of samples across clients by moving excess samples from over-provisioned
       clients to a common pool, and then distributing them to under-provisioned clients.
    This approach preserves the non-IID label skew better than re-shuffling all data.
    """
    # 1. Standard Dirichlet partition
    user_groups = partition_data_non_iid(labels, num_clients, beta, rng)
    
    # 2. Balance clients to have a target number of samples
    target_samples_per_client = len(labels) // num_clients
    
    # Identify over and under-provisioned clients
    over_clients = [i for i, indices in user_groups.items() if len(indices) > target_samples_per_client]
    under_clients = [i for i, indices in user_groups.items() if len(indices) < target_samples_per_client]
    
    # Create a pool of excess indices
    excess_pool = []
    for client_id in over_clients:
        excess_count = len(user_groups[client_id]) - target_samples_per_client
        
        # Take excess samples and add them to the pool
        excess_indices = user_groups[client_id][:excess_count]
        excess_pool.extend(excess_indices)
        
        # Remove them from the original client
        user_groups[client_id] = user_groups[client_id][excess_count:]

    rng.shuffle(excess_pool)
    
    # Distribute excess indices to under-provisioned clients
    for client_id in under_clients:
        needed_count = target_samples_per_client - len(user_groups[client_id])
        
        # Take needed samples from the pool
        if needed_count > 0 and len(excess_pool) > 0:
            take_count = min(needed_count, len(excess_pool))
            user_groups[client_id].extend(excess_pool[:take_count])
            excess_pool = excess_pool[take_count:]

    # If any samples remain in the pool (due to rounding), distribute them
    client_ids = list(range(num_clients))
    rng.shuffle(client_ids)
    idx = 0
    while len(excess_pool) > 0:
        client_id = client_ids[idx % num_clients]
        user_groups[client_id].append(excess_pool.pop())
        idx += 1
        
    return user_groups


def partition_data_iid_quantity_skew(labels, num_clients, min_size, max_size, rng: Any):
    """
    Partitions data with IID label distribution but varying quantity per client.
    """
    num_samples = len(labels)
    indices = np.arange(num_samples)
    rng.shuffle(indices)

    # Generate random sizes for each client
    client_sizes = rng.integers(min_size, max_size, num_clients)
    
    # Scale sizes to sum to the total number of samples
    total_requested_size = client_sizes.sum()
    scaled_sizes = (client_sizes / total_requested_size * num_samples).astype(int)
    
    # Adjust for rounding errors
    diff = num_samples - scaled_sizes.sum()
    scaled_sizes[-1] += diff

    user_groups = {}
    start_idx = 0
    for i in range(num_clients):
        end_idx = start_idx + scaled_sizes[i]
        user_groups[i] = indices[start_idx:end_idx]
        start_idx = end_idx
        
    return user_groups


def plot_partition_stats(user_groups, labels, num_clients, partition_type, log_dir):
    """Plots and saves the data distribution statistics for each client."""
    num_classes = len(np.unique(labels))
    client_distributions = np.zeros((num_clients, num_classes))

    for client_id, indices in user_groups.items():
        client_labels = labels[indices]
        for c in range(num_classes):
            client_distributions[client_id, c] = (client_labels == c).sum()
    
    fig, ax = plt.subplots(figsize=(12, 8))
    bottom = np.zeros(num_clients)
    
    for c in range(num_classes):
        ax.bar(range(num_clients), client_distributions[:, c], bottom=bottom, label=f'Class {c}')
        bottom += client_distributions[:, c]

    ax.set_xlabel('Client ID')
    ax.set_ylabel('Number of Samples')
    ax.set_title(f'Data Distribution Across Clients ({partition_type})')
    ax.legend()
    plt.tight_layout()
    
    plot_dir = log_dir
    create_folder_if_not_exists(plot_dir)
    plot_path = os.path.join(plot_dir, f'partition_{partition_type}.png')
    plt.savefig(plot_path)
    logger.info(f"Saved partition stats plot to {plot_path}")
    plt.close()


def record_data_statistic(client_data_loaders, num_clients):
    """Record statistics about the data distribution among clients."""
    for i in range(num_clients):
        train_dl, _ = client_data_loaders[i]
        
        # This is a more robust way to get labels from a Subset
        subset = train_dl.dataset
        if isinstance(subset, data.Subset):
            # Access targets from the original dataset via the subset's indices
            if hasattr(subset.dataset, 'targets'):
                targets = np.array(subset.dataset.targets)[subset.indices]
            elif hasattr(subset.dataset, 'labels'):
                targets = np.array(subset.dataset.labels)[subset.indices]
            else:
                # Try to get targets for each index
                targets = []
                for idx in subset.indices:
                    _, target = subset.dataset[idx]
                    targets.append(target)
                targets = np.array(targets)
        else:
            # Fallback for regular datasets
            if hasattr(subset, 'targets'):
                targets = np.array(subset.targets)
            elif hasattr(subset, 'labels'):
                targets = np.array(subset.labels)
            else:
                # Try to get targets by iterating
                targets = []
                for _, target in subset:
                    targets.append(target)
                targets = np.array(targets)

        # Count labels efficiently using numpy or collections
        label_count = defaultdict(int)
        unique_labels, counts = np.unique(targets, return_counts=True)
        for label, count in zip(unique_labels, counts):
            label_count[label] = count
            
        logger.info(f"Client {i} - total training samples: {len(subset)}")
        # Sort by key for consistent log output
        logger.info(f"Client {i} - training label distribution: {dict(sorted(label_count.items()))}")


def build_client_stats(
    *,
    dataset_name: str,
    num_clients: int,
    partition_type: str,
    partition_beta: float,
    seed: int,
    data_balance_strategy: str,
    imbalance_percentage: float,
    user_groups: Dict[int, List[int]],
    labels: np.ndarray,
) -> Dict[str, Any]:
    sample_counts: List[int] = []
    label_totals: Dict[int, int] = defaultdict(int)
    clients: Dict[str, Any] = {}

    for client_id in range(num_clients):
        indices = np.array(user_groups.get(client_id, []), dtype=np.int64)
        sample_counts.append(int(indices.size))

        if indices.size == 0:
            label_counts: Dict[str, int] = {}
        else:
            client_labels = labels[indices]
            unique_labels, counts = np.unique(client_labels, return_counts=True)
            label_counts = {str(int(label)): int(count) for label, count in zip(unique_labels, counts)}
            for label, count in zip(unique_labels, counts):
                label_totals[int(label)] += int(count)

        clients[str(client_id)] = {
            "num_samples": int(indices.size),
            "label_counts": label_counts,
        }

    stats = {
        "dataset": dataset_name,
        "seed": int(seed),
        "num_clients": int(num_clients),
        "partition": {
            "type": partition_type,
            "beta": float(partition_beta),
        },
        "data_balance_strategy": data_balance_strategy,
        "imbalance_percentage": float(imbalance_percentage),
        "summary": {
            "min_samples_per_client": min(sample_counts) if sample_counts else 0,
            "max_samples_per_client": max(sample_counts) if sample_counts else 0,
            "mean_samples_per_client": float(np.mean(sample_counts)) if sample_counts else 0.0,
            "label_totals": {str(k): int(v) for k, v in sorted(label_totals.items(), key=lambda kv: kv[0])},
        },
        "clients": clients,
    }
    return stats


def write_client_stats(stats: Dict[str, Any], path: str) -> None:
    dir_name = os.path.dirname(path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2, sort_keys=True)
    logger.info("Saved client stats to %s", path)


def get_client_data(
    dataset: str,
    dir_data: str,
    num_clients: int,
    partition_type: str,
    partition_beta: float = 0.5,
) -> Dict[int, Tuple[data.Dataset, data.Dataset]]:

    if dataset == "mnist":
        train_dataset, test_dataset = load_data_mnist(dir_data)
    elif dataset == "fmnist":
        train_dataset, test_dataset = load_data_fmnist(dir_data)
    elif dataset == "cifar10":
        train_dataset, test_dataset = load_data_cifar10(dir_data)
    elif dataset == "svhn":
        train_dataset, test_dataset = load_data_svhn(dir_data)
    elif dataset == "emnist_byclass":
        train_dataset, test_dataset = load_data_emnist_byclass(dir_data)
    elif dataset == "emnist_bymerge":
        train_dataset, test_dataset = load_data_emnist_bymerge(dir_data)
    else:
        raise ValueError("Dataset not supported")

    y_train = train_dataset.targets
    if y_train is not None:
        num_samples = len(y_train)
        num_classes = len(np.unique(y_train))
    else:
        raise ValueError("Cannot acqurie the number of samples and classes")

    if partition_type == "iid":
        logger.info("partitioning the data into IID setting")
        num_items_per_client = num_samples // num_clients
        remain = num_samples % num_clients
        lengths = [num_items_per_client] * num_clients
        lengths[-1] += remain
        client_datasets = data.random_split(train_dataset, lengths)
        client_datasets = {
            i: (client_datasets[i], test_dataset) for i in range(num_clients)
        }

    elif partition_type == "noniid":
        logger.info("partitioning the data into non-IID setting")

        class_indices = {
            i: np.where(np.array(train_dataset.targets) == i)[0]
            for i in range(num_classes)
        }
        client_indices = {cid: [] for cid in range(num_clients)}

        for c in range(num_classes):
            proportions = np.random.dirichlet(np.repeat(partition_beta, num_clients))
            proportions = proportions / proportions.sum() * len(class_indices[c])
            proportions = np.cumsum(proportions).astype(int)
            splits = np.split(class_indices[c], proportions[:-1])
            for cid in range(num_clients):
                client_indices[cid].extend(splits[cid].tolist())

        client_datasets = {
            cid: (get_full_subset(train_dataset, indices), test_dataset)
            for cid, indices in client_indices.items()
        }
    else:
        raise ValueError("Partition type not supported")

    record_data_statistic(client_datasets, num_clients)

    return client_datasets


def get_global_test_data_loader(
    dataset: str,
    dir_data: str,
    batch_size: int,
    seed: int = 42,  # Add seed parameter with default value
    num_workers: int = 0,  # Default to 0 for deterministic behavior
) -> data.DataLoader:
    if dataset == "mnist":
        _dir = os.path.join(dir_data, "mnist/")
        _, test_dataset = load_data_mnist(_dir)
    elif dataset == "fmnist":
        _dir = os.path.join(dir_data, "fmnist/")
        _, test_dataset = load_data_fmnist(_dir)
    elif dataset == "cifar10":
        _dir = os.path.join(dir_data, "cifar10/")
        _, test_dataset = load_data_cifar10(_dir)
    elif dataset == "svhn":
        _dir = os.path.join(dir_data, "svhn/")
        _, test_dataset = load_data_svhn(_dir)
    elif dataset == "emnist_byclass":
        _dir = os.path.join(dir_data, "emnist/")
        _, test_dataset = load_data_emnist_byclass(_dir)
    elif dataset == "emnist_bymerge":
        _dir = os.path.join(dir_data, "emnist/")
        _, test_dataset = load_data_emnist_bymerge(_dir)
    elif dataset == "tinyimagenet":
        _dir = dir_data
        _, test_dataset = load_data_tinyimagenet(_dir)
    else:
        raise ValueError("Dataset not supported")

    # Set seed for reproducibility
    g_test = torch.Generator()
    g_test.manual_seed(seed)
    
    test_dl = create_data_loader(
        dataset=test_dataset, 
        batch_size=batch_size,
        shuffle=False,  # Usually don't shuffle test data
        num_workers=num_workers,
        generator=g_test
    )
    return test_dl


def get_full_subset(dataset, indices):
    class SubDataset(data.Dataset):
        def __init__(self, dataset, indices):
            self.dataset = dataset
            self.indices = indices
            self.targets = [dataset.targets[i] for i in indices]
            self.classes = getattr(dataset, "classes", None)

        def __getitem__(self, idx):
            # data, _ = self.dataset[self.indices[idx]]
            # return data, self.targets[idx]
            return self.dataset[self.indices[idx]]

        def __len__(self):
            return len(self.indices)

    return SubDataset(dataset, indices)


def get_client_data_loader(
    dataset_name: str,
    dir_data: str,
    num_clients: int,
    partition_type: str,
    partition_dirichlet_beta: float,
    batch_size: int,
    seed: int,
    num_workers: int = 0,  # Default to 0
    data_balance_strategy: str = "balanced",
    imbalance_percentage: float = 0.1,
    plot_partition_stats_flag: bool = False,
    log_dir: str = './logs',
    min_samples_per_client: int = 100,
    max_samples_per_client: int = 2000,
    use_fake_data: bool = False,
    fake_train_size: int = 2000,
    fake_test_size: int = 500,
    strict_data: bool = False,
    client_stats_path: str | None = None,
) -> dict:
    """
    return the dataloaders for clients
    """
    logger.info("start data partitioning...")

    rng = np.random.default_rng(seed)
    train_dataset, test_dataset = get_dataset(
        dataset_name,
        dir_data,
        seed=seed,
        use_fake_data=use_fake_data,
        fake_train_size=fake_train_size,
        fake_test_size=fake_test_size,
        allow_fallback=not strict_data,
    )

    if data_balance_strategy == "extreme_imbalance":
        # This function now correctly handles partitioning for other clients internally
        train_ds, test_ds, user_groups = partition_data_imbalance(
            train_dataset,
            test_dataset,
            num_clients,
            partition_type,
            partition_dirichlet_beta,
            imbalance_percentage,
            min_samples_per_client,
            max_samples_per_client,
            rng,
        )
    else:
        # Handle "balanced" and "soft_imbalance"
        dataset_to_partition = train_dataset
        if data_balance_strategy == "soft_imbalance":
            logger.info("Applying soft imbalance: removing 90% of class 0 samples.")
            targets = np.array(train_dataset.targets)
            class0_indices = np.where(targets == 0)[0]
            other_indices = np.where(targets != 0)[0]
            
            # Keep 10% of class 0
            rng.shuffle(class0_indices)
            keep_count = int(len(class0_indices) * 0.1)
            kept_class0_indices = class0_indices[:keep_count]
            
            # Combine indices and create a subset
            final_indices = np.concatenate((kept_class0_indices, other_indices))
            rng.shuffle(final_indices)
            
            dataset_to_partition = torch.utils.data.Subset(train_dataset, final_indices)
            dataset_to_partition.targets = targets[final_indices]
        
        # Partition the prepared dataset (either full or subset)
        train_ds, test_ds, groups_on_partitioned_ds = partition_data(
            dataset_to_partition,
            test_dataset,
            num_clients,
            partition_type,
            partition_dirichlet_beta,
            min_samples_per_client,
            max_samples_per_client,
            rng,
        )

        # Map indices from the partitioned dataset back to the original full dataset
        user_groups = {}
        original_indices_map = getattr(dataset_to_partition, 'indices', list(range(len(dataset_to_partition))))
        for client_id, subset_indices in groups_on_partitioned_ds.items():
            user_groups[client_id] = [original_indices_map[i] for i in subset_indices]

    if plot_partition_stats_flag:
        labels = np.array(train_dataset.targets) # Use original labels for stats
        plot_partition_stats(
            user_groups, labels, num_clients, f"{partition_type}_{data_balance_strategy}", log_dir
        )

    labels = np.array(train_dataset.targets)
    stats = build_client_stats(
        dataset_name=dataset_name,
        num_clients=num_clients,
        partition_type=partition_type,
        partition_beta=partition_dirichlet_beta,
        seed=seed,
        data_balance_strategy=data_balance_strategy,
        imbalance_percentage=imbalance_percentage,
        user_groups=user_groups,
        labels=labels,
    )
    if client_stats_path:
        write_client_stats(stats, client_stats_path)
    
    client_loaders = {}
    
    # Create one generator for all test DataLoaders (they are identical)
    g_test = torch.Generator()
    g_test.manual_seed(seed)
    test_dl = create_data_loader(test_dataset, batch_size, False, num_workers, g_test)

    for i in range(num_clients):
        user_indices = user_groups.get(i, [])
        train_ds_client = data.Subset(train_dataset, user_indices)

        # Create a unique, deterministic generator for each client's training data
        g_train = torch.Generator()
        g_train.manual_seed(seed + i)

        # Correctly call the enhanced create_data_loader
        train_dl = create_data_loader(train_ds_client, batch_size, True, num_workers, g_train)
        
        # All clients share the same test dataloader instance
        client_loaders[i] = (train_dl, test_dl)

    record_data_statistic(client_loaders, num_clients)
    return client_loaders


def create_data_loader(
    dataset: data.Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    generator: torch.Generator,
):
    """Creates a DataLoader with reproducibility settings."""
    return data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=num_workers,
        generator=generator,
    )


def partition_data_imbalance(
    train_ds,
    test_ds,
    num_clients: int,
    partition_type: str,
    partition_dirichlet_beta: float,
    imbalance_percentage: float = 0.1,
    min_samples_per_client: int = 100,
    max_samples_per_client: int = 2000,
    rng: Any = None,
) -> tuple:
    """
    Partitions data for an imbalanced scenario. Client 0 gets a small portion of class 0.
    The remaining data (from other classes only) is partitioned among the other N-1 clients.
    """
    if rng is None:
        rng = np.random.default_rng()

    y_train = np.array(train_ds.targets)
    
    # Get indices for class 0 and other classes
    class0_indices = np.where(y_train == 0)[0]
    other_classes_indices = np.where(y_train != 0)[0]
    
    # Shuffle class 0 indices
    rng.shuffle(class0_indices)
    
    # Determine how many class 0 samples to give to client 0
    num_class0_for_client0 = int(len(class0_indices) * imbalance_percentage)
    client0_indices = class0_indices[:num_class0_for_client0]
    
    user_groups = {0: client0_indices}
    
    # The rest of the clients get data ONLY from other classes.
    # The remaining class 0 data is effectively discarded.
    remaining_indices = other_classes_indices
    rng.shuffle(remaining_indices)
    
    if num_clients > 1:
        # Create a subset of the dataset for the remaining clients (contains no class 0)
        remaining_dataset = data.Subset(train_ds, remaining_indices)
        remaining_dataset.targets = y_train[remaining_indices]

        # Partition this remaining data among the other clients
        _, _, other_clients_groups = partition_data(
            remaining_dataset,
            test_ds,
            num_clients - 1,
            partition_type,
            partition_dirichlet_beta,
            min_samples_per_client,
            max_samples_per_client,
            rng,
        )

        # Map the indices from the subset back to the original dataset indices
        # and adjust client IDs (from 0..N-2 to 1..N-1)
        for other_client_id, subset_indices in other_clients_groups.items():
            original_dataset_indices = remaining_indices[subset_indices]
            user_groups[other_client_id + 1] = original_dataset_indices
            
    return train_ds, test_ds, user_groups
