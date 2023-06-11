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

    def __init__(
        self, root_dir=PROJECT_ROOT, augmentation=False, mask=True, resolution=128
    ):
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
        self.file_names = self.get_file_names(self.mask_pairs)

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

    @staticmethod
    def get_file_names(mask_pairs):
        """
        Use the pairs of directories for mask and image data to recursively obtain the
        paths of all .nii files
        The img-mask pairs always contain a directory for each measurement, which in turn
        contain a single directory in which the .nii file is contained.
        """
        for pair in mask_pairs:
            pair["img_files"] = glob.glob(
                os.path.join(pair["img"], "*/*/*.nii"), recursive=True
            )
            pair["mask_files"] = glob.glob(
                os.path.join(pair["mask"], "*/*/*.nii"), recursive=True
            )
            if len(pair["img_files"]) != len(pair["mask_files"]):
                mask_pairs.remove(pair)
                Warning(
                    f"Uneven Amount of Measurements in {pair}; removed from mask_pairs."
                )
        # now finally obtain a paired list with each individual .nii image-mask pair
        file_names = [
            {"image": img, "mask": msk}  # construct dict from nested list
            for img, msk in zip(pair["img_files"], pair["mask_files"])
            for pair in mask_pairs
        ]
        return file_names

    @staticmethod
    def mask_image(image, mask):
        """
        Crop image according to mask
        """
        # axes of mask and image are swapped for some reason
        mask_swapped = np.swapaxes(mask, 0, 2)  # this is correct
        # need to rotate for some reason TODO: write tests to verify that all adni images are processed correctly.
        mask_rot90 = np.rot90(mask_swapped, 2)
        masked_image = np.where(mask_rot90 > 0, image, 0)
        return masked_image

    @staticmethod
    def crop_image(image):
        crop_mask = image > 0
        # coords of nonblack pixels
        coords = np.argwhere(crop_mask)
        # Bounding box of non-black pixels.
        x0, y0, z0 = coords.min(axis=0)
        x1, y1, z1 = coords.max(axis=0) + 1  # slices are exclusive at the top
        cropped = image[x0:x1, y0:y1, z0:z1]
        # add channel dimension to image
        padded_crop = tio.CropOrPad(np.max(cropped.shape))(cropped.copy()[None])
        padded_crop = np.transpose(padded_crop, (1, 2, 3, 0))
        return padded_crop

    def __getitem__(self, index) -> Any:
        """
        Get a single image that is cropped and preprocessed
        """
        res = self.resolution
        entry = self.file_names[index]
        img = nib.load(entry["image"])
        msk = nib.load(entry["mask"])
        # get np arrays
        img_data = np.asanyarray(img.dataobj)  # TODO: add non masking case
        msk_data = np.asanyarray(msk.dataobj)
        # mask then crop
        masked_image = self.mask_image(img_data, msk_data)
        cropped_image = crop_image(masked_image)
        # resize to uniform size
        img = resize(cropped_image, (res, res, res), mode="constant")
        if self.augmentation:
            random_n = torch.rand(1)
            random_i = 0.3 * torch.rand(1)[0] + 0.7
            if random_n[0] > 0.5:
                img = np.flip(img, 0)
            img = img * random_i.data.cpu().numpy()
        out = torch.from_numpy(img).float().view(1, res, res, res)
        # very sussy line
        out = out * 2 - 1
        return {"data", out}


# TODO: move to test file to verify image loading
img_slice = img_data[:, :, 85]
msk_slice = msk_data[:, :, 85]
msk_slice_swapped = msk_swapped[:, :, 85]
msk_slice_rot90 = msk_rot90[:, :, 85]

# check if mask and image match
masked_slice = np.where(msk_slice_rot90 > 0, img_slice, 0)
# this seems to be correct
test = torch.squeeze(out)
test = test[:, :, 64]
test = test * 2 - 1
plt.imshow(test)
plt.show()

# only select images with a high resolution
# 256 is our target, may need to downsample for vqgan
