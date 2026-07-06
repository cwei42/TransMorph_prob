"""
Generic paired-registration dataset for TM-prob.

Institution-specific file discovery has been replaced with a manifest file:
a JSON list of dicts, each describing one fixed/moving image pair, e.g.

    [
      {
        "fixed_image": "path/to/case001/fixed_image.mha",
        "moving_image": "path/to/case001/moving_image.mha",
        "fixed_label": "path/to/case001/fixed_label.mha",
        "moving_label": "path/to/case001/moving_label.mha",
        "fixed_dose": "path/to/case001/fixed_dose.mha",       # optional, for dose-uncertainty propagation
        "moving_dose": "path/to/case001/moving_dose.mha",     # optional
        "fixed_PTV": "path/to/case001/fixed_PTV.mha",         # optional, binary mask
        "moving_PTV": "path/to/case001/moving_PTV.mha"        # optional, binary mask
      },
      ...
    ]

Labels are expected as a single integer-encoded segmentation volume (0 =
background, 1..8 = organs, matching the class order in `model.py`'s
`sample_contours`: Lung_Left, Lung_Right, Spinal_canal, Trachea, Aorta,
Esophagus, Heart, GTV).

Landmark-based TRE evaluation (in `TMProb.validation_step`) looks for a CSV
file at `path/to/case001/landmarks_new.csv`
containing moving/fixed landmark voxel coordinates; point to your own
landmark file with that naming convention, or adapt the path in `model.py`.
"""
from __future__ import annotations

import collections.abc
import json
from collections.abc import Sequence

import numpy as np
import SimpleITK as sitk
from torch.utils.data import Subset
from monai.data import Dataset
from monai.transforms import (
    Compose,
    LoadImaged,
    ScaleIntensityRangePercentilesd,
    ResizeWithPadOrCropd,
    Spacingd,
    RandGaussianNoised,
)


class PairedRegistrationDataset(Dataset):
    """MONAI `Dataset` that also returns per-item image metadata (origin/spacing/
    direction), which `model.py` uses to write predictions back out in the
    original image geometry."""

    def __getitem__(self, index: int | slice | Sequence[int]):
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            return Subset(dataset=self, indices=range(start, stop, step))
        if isinstance(index, collections.abc.Sequence):
            return Subset(dataset=self, indices=index)

        data_i = self.data[index]
        ref_key = "fixed_label" if "fixed_label" in data_i else "fixed_image"
        sitk_img = sitk.ReadImage(data_i[ref_key])
        metadata = {
            "path": data_i[ref_key],
            "origin": np.asarray(sitk_img.GetOrigin()),
            "direction": np.asarray(sitk_img.GetDirection()),
            "spacing": np.asarray(sitk_img.GetSpacing()),
            "size": np.asarray(sitk_img.GetSize()),
        }
        return self._transform(index), metadata


def _load_manifest(manifest_path):
    with open(manifest_path) as f:
        return json.load(f)


_ALL_KEYS = ["fixed_image", "moving_image", "fixed_label", "moving_label",
             "fixed_dose", "moving_dose", "fixed_PTV", "moving_PTV"]


class RegistrationData:
    """Builds train/val `PairedRegistrationDataset`s from JSON manifests.

    Args:
        train_manifest: path to a JSON manifest (see module docstring) used for
            training, or None to skip.
        val_manifest: path to a JSON manifest used for validation, or None.
        spatial_size: target volume size after center pad/crop, (H, W, D).
        pixdim: optional target voxel spacing (sx, sy, sz) in mm; resamples if given.
        augment: whether to apply light training-time augmentation (Gaussian noise).
    """

    def __init__(self, train_manifest=None, val_manifest=None,
                 spatial_size=(320, 224, 160), pixdim=None, augment=True):
        self.train_files = _load_manifest(train_manifest) if train_manifest else []
        self.val_files = _load_manifest(val_manifest) if val_manifest else []

        def present_keys(files):
            return sorted(set().union(*[set(d.keys()) for d in files])) if files else _ALL_KEYS[:4]

        train_keys = present_keys(self.train_files)
        val_keys = present_keys(self.val_files)

        train_steps = [
            LoadImaged(keys=train_keys, ensure_channel_first=True, reader="ITKReader", allow_missing_keys=True),
            ScaleIntensityRangePercentilesd(keys=["fixed_image", "moving_image"], lower=0, upper=99.5,
                                             b_min=0, b_max=1, clip=True),
        ]
        val_steps = [
            LoadImaged(keys=val_keys, ensure_channel_first=True, reader="ITKReader", allow_missing_keys=True),
            ScaleIntensityRangePercentilesd(keys=["fixed_image", "moving_image"], lower=0, upper=99.5,
                                             b_min=0, b_max=1, clip=True),
        ]
        if pixdim is not None:
            resample_keys = [k for k in train_keys if k != "fixed_PTV" and k != "moving_PTV"]
            train_steps.append(Spacingd(keys=resample_keys, pixdim=pixdim, mode="nearest"))
            val_steps.append(Spacingd(keys=[k for k in val_keys if k not in ("fixed_PTV", "moving_PTV")],
                                       pixdim=pixdim, mode="nearest"))
        train_steps.append(ResizeWithPadOrCropd(keys=train_keys, spatial_size=spatial_size,
                                                 method="symmetric", mode="constant"))
        val_steps.append(ResizeWithPadOrCropd(keys=val_keys, spatial_size=spatial_size,
                                               method="symmetric", mode="constant"))
        if augment:
            train_steps.append(RandGaussianNoised(keys=["fixed_image", "moving_image"], prob=0.6))

        self.train_transforms = Compose(train_steps)
        self.val_transforms = Compose(val_steps)

    def loaddata(self):
        train_ds = PairedRegistrationDataset(data=self.train_files, transform=self.train_transforms)
        val_ds = PairedRegistrationDataset(data=self.val_files, transform=self.val_transforms)
        return train_ds, val_ds
