from synthraddiffusion.dataset import DEFAULTDataset, ADNIDataset, SYNTHRADDataset
from torch.utils.data import WeightedRandomSampler


def get_dataset(config):
    """wrapper to unify data loading from ADNI and Synthrad"""
