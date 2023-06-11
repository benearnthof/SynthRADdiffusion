from synthraddiffusion.dataset import DEFAULTDataset, ADNIDataset, SYNTHRADDataset
from torch.utils.data import WeightedRandomSampler


def get_dataset(config):
    """wrapper to unify data loading from ADNI and Synthrad"""
    if config.dataset.name == "DEFAULT":
        train_dataset = DEFAULTDataset(root_dir=config.dataset.root_dir)
        val_dataset = DEFAULTDataset(root_dir=config.dataset.root_dir)
        sampler = None

    if config.dataset.name == "ADNI":
        train_dataset = ADNIDataset(root_dir=config.dataset.root_dir, augmentation=True)
        val_dataset = ADNIDataset(root_dir=config.dataset.root_dir, augmentation=True)
        sampler = None
        return train_dataset, val_dataset, sampler

    if config.dataset.name == "SYNTHRAD":
        train_dataset = ADNIDataset(root_dir=config.dataset.root_dir, augmentation=True)
        val_dataset = ADNIDataset(root_dir=config.dataset.root_dir, augmentation=True)
        sampler = None
        return train_dataset, val_dataset, sampler

    raise ValueError(f"{config.dataset.name} Dataset is not available.")
