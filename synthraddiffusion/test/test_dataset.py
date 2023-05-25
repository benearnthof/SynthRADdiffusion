"""Test Functionality of Dataset Classes"""

from synthraddiffusion.dataset.adni import ADNIDataset
from pathlib import Path

# Test Object Creation
# GIVEN
ds = ADNIDataset()
# WHEN
# THEN
assert ds is not None
assert (
    len(ds.subject_folders) == 1254
)  # all subjects that were measured with 3T machines
assert len(ds.mask_pairs) == 287
assert len(ds.mask_pairs[0]) == 4  # 4 images for 1 subject
assert isinstance(ds.file_names, list)
assert isinstance(ds.file_names[0], dict)

# check that all files are paired
# GIVEN
ds = ADNIDataset()
# WHEN
filenamelen = [len(x) for x in ds.file_names]
assert all(i == 2 for i in filenamelen)

# check all files exist
check = [Path(x["image"]).exists() and Path(x["mask"]).exists() for x in ds.file_names]
assert all(x for x in check)

# check that pairs have the same root
# need to check 4 parents down since the files are ordered in this way:
# root/MASK_OR_IMAGE/date_of_measurement/foldernum/*nii
check = [
    Path(x["image"]).parent.parent.parent.parent
    == Path(x["mask"]).parent.parent.parent.parent
    for x in ds.file_names
]
assert all(x for x in check)
