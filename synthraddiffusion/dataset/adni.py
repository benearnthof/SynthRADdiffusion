"""
Functions to import ADNI data for pretraining MRI VQGAN
"""

# Are all files of the same reolution?
# are they of the same resolution as the SynthRAD data?
# How many images do we need? 10k should be enough, 100k would be better
# masks are already available for each respective measurement
# how do the scaled and unscaled images compare? (unscaled, scaled, scaled_2)

# Synthrad data specifications
# T1-weighted gradient echo or inversion prepared turbo field echo
# B and C were acquired with contrast agent
# A were acquired without contrast agent

# Center A: 240-1024 Rows & Columns
# Center B: 236 Rows, 174-236 Columns
# Center C: 224-256 Rows, 204-256 Columns

from typing import List
import csv
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import os
from torchvision import transforms
from skimage.transform import resize
from nilearn import surface
import nibabel as nib
import argparse
import glob
import torchio as tio
from pathlib import Path

PROJECT_ROOT = Path("D:\synthRAD2023")

MASK_SUFFIXES = [
    "MPR-R____Mask",
    "MPR____Mask",
    "MPR__GradWarp__B1_Correction__Mask",
    "MPR-R__GradWarp__B1_Correction__Mask",
]

IMG_SUFFIXES = [
    "MPR-R____N3",
    "MPR____N3",
    "MPR__GradWarp__B1_Correction",
    "MPR-R__GradWarp__B1_Correction",
]


class ADNIDataset(Dataset):
    """
    Wrapper around PyTorch Dataset to load and transform ADNI data
    """

    def __init__(self, root_dir=PROJECT_ROOT, augmentation=False):
        self.root_dir = root_dir
        self.subject_folders = glob.glob(
            os.path.join(root_dir, "ADNI_3T/ADNI/*"), recursive=True
        )
        self.augmentation = augmentation

    def __len__(self):
        return len(self.subject_folders)

    def get_maskpairs(dirlist: List[List[str]]):
        """
        check if dirlist contains pairs of image- and respective mask folders
        and return a list of image-mask directory pairs
            dirlist: Nested List of strings containing all subdirectories in each respective subject folder
        """
        pairs = []
        for dirs in dirlist:
            # folders containing masks
            mask_matches = [
                d for d in dirs if any(xs in Path(d).name for xs in MASK_SUFFIXES)
            ]
            # proceed only if mask_matches is not empty
            if not mask_matches:
                continue
            img_matches = []
            for entry in mask_matches:
                if Path(entry).name == "MPR-R____Mask":
                    img_matches.append(Path(entry).parent / "MPR-R____N3")
                elif Path(entry).name == "MPR____Mask":
                    img_matches.append(Path(entry).parent / "MPR____N3")
                elif Path(entry).name == "MPR__GradWarp__B1_Correction__Mask":
                    img_matches.append(
                        Path(entry).parent / "MPR__GradWarp__B1_Correction"
                    )
                elif Path(entry).name == "MPR-R__GradWarp__B1_Correction__Mask":
                    img_matches.append(
                        Path(entry).parent / "MPR-R__GradWarp__B1_Correction"
                    )
                else:
                    Warning(f"Did not find a match for folder {entry}")
            if len(mask_matches) == len(img_matches):
                # construct pair of image and mask file folder
                pairdict = [
                    {"img": i, "mask": Path(m)}
                    for i, m in zip(img_matches, mask_matches)
                ]
                pairs.extend(pairdict)
        # check if all pair folders exist
        check = [pair["img"].exists() and pair["mask"].exists() for pair in pairs]
        output = [pair for pair, c in zip(pairs, check) if c]
        return output

    def get_maskfolders(self):
        """
        get all unique folders that contain both an image and a mask
        """
        folders = self.subject_folders
        # list of all subdirectories contained in subject folders
        subdirs = [glob.glob(os.path.join(folder, "*")) for folder in folders]
        # get folders with images and folders containing their respective masks
        # vector that indicates if subject_folder contains an image-mask pair
        maskpairs = get_maskpairs(subdirs)

    # only select images that have a corresponding mask we can use
    # only select images with a high resolution
    # 256 if possible 512
