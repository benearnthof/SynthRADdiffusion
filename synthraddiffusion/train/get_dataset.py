from synthraddiffusion.dataset import DEFAULTDataset, ADNIDataset, SYNTHRADDataset
from torch.utils.data import WeightedRandomSampler


def get_dataset(config):
    """wrapper to unify data loading from ADNI and Synthrad"""
    if config.datasets.name == "DEFAULT":
        train_dataset = DEFAULTDataset(root_dir=config.datasets.root_dir)
        val_dataset = DEFAULTDataset(root_dir=config.datasets.root_dir)
        sampler = None

    if config.datasets.name == "ADNI":
        train_dataset = ADNIDataset(
            root_dir=config.datasets.root_dir, augmentation=True
        )
        val_dataset = ADNIDataset(root_dir=config.datasets.root_dir, augmentation=True)
        sampler = None
        return train_dataset, val_dataset, sampler

    if config.datasets.name == "SYNTHRAD":
        train_dataset = ADNIDataset(
            root_dir=config.datasets.root_dir, augmentation=True
        )
        val_dataset = ADNIDataset(root_dir=config.datasets.root_dir, augmentation=True)
        sampler = None
        return train_dataset, val_dataset, sampler

    raise ValueError(f"{config.datasets.name} Dataset is not available.")
