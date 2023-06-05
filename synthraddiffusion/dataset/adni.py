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

from typing import List, Any
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
from warnings import warn

PROJECT_ROOT = Path(
    "D:\synthRAD2023"
)  # TODO: remove as the dataset is instantiated only through config

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
        # list of all subdirectories contained in subject folders
        self.subdirs = [
            glob.glob(os.path.join(folder, "*")) for folder in self.subject_folders
        ]
        # get folders with images and folders containing their respective masks
        # vector that indicates if subject_folder contains an image-mask pair
        self.mask_pairs = self.get_maskpairs(self.subdirs)
        self.file_names = self.get_file_names()

    def __len__(self):
        return len(self.file_names)

    @staticmethod
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
                if Path(entry).name == MASK_SUFFIXES[0]:
                    img_matches.append(Path(entry).parent / IMG_SUFFIXES[0])
                elif Path(entry).name == MASK_SUFFIXES[1]:
                    img_matches.append(Path(entry).parent / IMG_SUFFIXES[1])
                elif Path(entry).name == MASK_SUFFIXES[2]:
                    img_matches.append(Path(entry).parent / IMG_SUFFIXES[2])
                elif Path(entry).name == MASK_SUFFIXES[3]:
                    img_matches.append(Path(entry).parent / IMG_SUFFIXES[3])
                else:
                    warn(f"Did not find a match for folder {entry}")
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

    def get_file_names(self):
        """
        Use the pairs of directories for mask and image data to recursively obtain the
        paths of all .nii files
        The img-mask pairs always contain a directory for each measurement, which in turn
        contain a single directory in which the .nii file is contained.
        """
        for pair in self.mask_pairs:
            pair["img_files"] = glob.glob(
                os.path.join(pair["img"], "*/*/*.nii"), recursive=True
            )
            pair["mask_files"] = glob.glob(
                os.path.join(pair["mask"], "*/*/*.nii"), recursive=True
            )
            if len(pair["img_files"]) != len(pair["mask_files"]):
                self.mask_pairs.remove(pair)
                Warning(
                    f"Uneven Amount of Measurements in {pair}; removed from mask_pairs."
                )
        # now finally obtain a paired list with each individual .nii image-mask pair
        file_names = [
            {"image": img, "mask": msk}  # construct dict from nested list
            for img, msk in zip(pair["img_files"], pair["mask_files"])
            for pair in self.mask_pairs
        ]
        return file_names

    def crop(self, image, mask):
        """
        Crop image according to mask
        """
        pass

    def __getitem__(self, index) -> Any:
        """
        Get a single image that is cropped and preprocessed
        """
        entry = self.file_names[index]
        img = nib.load(entry["image"])
        msk = nib.load(entry["mask"])
        # axes of mask and image are swapped for some reason
        img2 = np.swapaxes(np.asanyarray(img.dataobj), 1, 2)
        # this is not correct, need to visualize what's going on and then align with mask

        img3 = np.flip(img2, 1)
        img4 = np.flip(img3, 2)

    # only select images with a high resolution
    # 256 if possible 512
