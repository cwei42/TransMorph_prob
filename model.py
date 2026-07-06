"""
TM-prob: TransMorph-based probabilistic deformable image registration.

Predicts a voxel-wise Gaussian distribution over dense displacement fields
(mean + variance) in a single forward pass, using a frequency-domain
Gaussian-process prior/reparameterization for spatially correlated
uncertainty. See ``README.md`` for the model description and paper
reference.

Modified by: 
Chengtao Wei
2026/07/26
"""
import os
import time

import numpy as np
from scipy.stats import pearsonr
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft
from torch import optim
import lightning as L
import SimpleITK as sitk

from monai.networks.blocks import Warp, DVF2DDF
from monai.networks.utils import one_hot
from monai.losses import BendingEnergyLoss, GlobalMutualInformationLoss, DiceLoss
from monai.metrics import DiceMetric, SSIMMetric

from models import TransMorph_bayesian_2
import models.configs_TransMorph as configs


class TMProb(L.LightningModule):
    """Variational Bayesian TransMorph registration model (TM-prob).

    Predicts voxel-wise mean displacement (DVF) and log-variance; the mean
    is integrated (scaling-and-squaring) into a diffeomorphic DDF, and a
    DDF sample is drawn via reparameterization with spatially correlated
    (frequency-domain filtered) noise. Trained with an MSE + bending-energy
    + Dice + frequency-domain-GP-KL objective.
    """

    def __init__(self, lr=1e-4, lMSE=10, lBE=1, ldice=0, kl=1e-2,
                 prior_lambda=150.0, correlation_length=8.0,
                 num_label_class=9, label_weight=None,
                 kl_warmup_epochs=70, savedir=None):
        super().__init__()
        self.save_hyperparameters()
        self.rootdir = savedir
        self.correlation_length = correlation_length
        self.prior_lambda = prior_lambda
        self.kl = kl
        self.kl_warmup_epochs = kl_warmup_epochs
        self.lr = lr
        self.lMSE = lMSE
        self.lBE = lBE
        self.ldice = ldice
        self.num_class = num_label_class

        config = configs.get_3DTransMorph_config()
        self.model = TransMorph_bayesian_2.TransMorph(config)
        self.dvf2ddf = DVF2DDF()

        # warp_layer_nn (nearest) is not differentiable and is only used for
        # warping label maps at inference time.
        self.warp_layer = Warp(mode="bilinear", padding_mode="border")
        self.warp_layer_nn = Warp(mode="nearest", padding_mode="border")

        self.MSE_loss = nn.MSELoss()
        self.BE = BendingEnergyLoss()
        self.MI_loss = GlobalMutualInformationLoss(num_bins=32)
        if label_weight is None:
            label_weight = torch.ones(num_label_class - 1)
        self.label_loss = DiceLoss(include_background=False, weight=label_weight)

        self.dice_metric = DiceMetric(include_background=False,
                                       reduction="mean_batch",
                                       get_not_nans=False,
                                       ignore_empty=True,
                                       num_classes=num_label_class)
        self.SSIM_metric = SSIMMetric(spatial_dims=3)

    def configure_optimizers(self):
        return optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=0, amsgrad=True)

    def forward(self, batch_data):
        """Predict a DDF sample and its distribution parameters.

        Returns:
            ddf: sampled dense displacement field, [B, 3, H, W, D]
            log_variance: predicted per-voxel log-variance, [B, 3, H, W, D]
            ddf_mu: mean (diffeomorphic) DDF, [B, 3, H, W, D]
            dvf_mu: mean DVF prior to scaling-and-squaring integration
        """
        fixed_image = batch_data["fixed_image"]
        moving_image = batch_data["moving_image"]

        dvf_mu, log_variance = self.model(torch.cat((moving_image, fixed_image), dim=1))
        ddf_mu = self.dvf2ddf(dvf_mu)

        noise = frequency_domain_sampling(log_variance, correlation_length=self.correlation_length)
        ddf = ddf_mu + noise

        return ddf, log_variance, ddf_mu, dvf_mu

    def training_step(self, batch, index):
        metadata = batch[1]
        batch = batch[0]

        fixed_image = batch["fixed_image"]
        fixed_label = batch["fixed_label"]
        moving_label = batch["moving_label"]
        moving_label_one_hot = one_hot(moving_label, num_classes=self.num_class)

        ddf, log_variance, ddf_mu, _ = self.forward(batch)
        pred_image = self.warp_layer(batch["moving_image"], ddf)

        kl_loss = kl_divergence_frequency_domain(log_variance, self.correlation_length,
                                                   prior_variance=self.prior_lambda)
        mse_loss = self.MSE_loss(pred_image, fixed_image)
        bend_energy = self.BE(ddf_mu)
        pred_label_one_hot = self.warp_layer(moving_label_one_hot, ddf_mu)
        dice = self.label_loss(pred_label_one_hot, one_hot(fixed_label, num_classes=self.num_class))

        beta = 0.0 if self.current_epoch < self.kl_warmup_epochs else 1.0
        loss = self.lMSE * mse_loss + self.lBE * bend_energy + beta * self.kl * kl_loss + self.ldice * dice

        self.log("train/loss", loss.item(), prog_bar=True, on_epoch=True)
        self.log_dict({
            "train/mseloss": mse_loss,
            "train/bendenergy": bend_energy,
            "train/diceloss": dice,
            "train/klloss": kl_loss,
        }, on_epoch=True, on_step=True)
        return loss

    def validation_step(self, batch, index):
        metadata = batch[1]
        batch = batch[0]
        path = metadata['path'][0].replace('/', '_').replace('.mha', '_')[1:].replace('original_', '')

        val_fixed_image = batch["fixed_image"]
        val_fixed_label = batch["fixed_label"]
        val_frac_image = batch["moving_image"]
        val_frac_label = batch["moving_label"]
        moving_label_one_hot = one_hot(val_frac_label, num_classes=self.num_class)

        ddf, log_variance, ddf_mu, dvf_mu = self.forward(batch)
        ddf_array = ddf_mu[0].detach().cpu().numpy()

        val_pred_image = self.warp_layer(val_frac_image, ddf_mu)
        val_pred_label_one_hot = self.warp_layer(moving_label_one_hot, ddf_mu)

        # Voxel-wise displacement uncertainty (mm), scaled by voxel spacing.
        sigma = torch.exp(log_variance / 2.0)
        sigma[:, 0] *= 1.5
        sigma[:, 1] *= 1.5
        sigma[:, 2] *= 3
        sigma_amplitude = torch.sqrt(torch.sum(sigma ** 2, dim=1, keepdim=True))
        sigma_amplitude_array = sigma_amplitude[0][0].detach().cpu().numpy()

        landmark_path = metadata['path'][0].replace("label_sum.mha", "landmarks_new.csv")
        tre_stats = None
        if os.path.exists(landmark_path):
            landmarks = np.transpose(np.genfromtxt(landmark_path, delimiter=',', dtype=np.float64), (1, 0))
            mov, fix = np.split(landmarks, 2)
            oridiff, sigmas, tres = transform_landmarks(fix, mov, ddf_array, sigma_amplitude_array,
                                                          self.rootdir, savepath=None, spacing=(1.5, 1.5, 3))
            tre_stats = {"TRE": np.mean(tres), "Pearson": pearsonr(sigmas, tres)[0]}
        else:
            print("No landmarks found for case", metadata['path'][0])

        val_fixed_label_one_hot = one_hot(val_fixed_label, num_classes=self.num_class)
        val_pred_label_one_hot = (val_pred_label_one_hot > 0.5).float()
        dice = self.dice_metric(y_pred=val_pred_label_one_hot, y=val_fixed_label_one_hot)[0].detach().cpu().numpy()
        ssim = self.SSIM_metric(y_pred=val_pred_image, y=val_fixed_image)[0].detach().cpu().numpy()

        det = compute_jacobian_determinant(ddf)
        num_folding = np.sum((det.detach().cpu().numpy() <= 0).astype(np.int16))
        per_folding = num_folding / det.numel() * 100

        # Uncertainty propagation to contours/doses for a subset of cases;
        # override `savelist` (or always run) as needed.
        savelist = getattr(self, "savelist", [])
        if index in savelist and self.rootdir is not None:
            val_fixed_dose = batch.get("fixed_dose")
            val_frac_dose = batch.get("moving_dose")
            val_fixed_PTV = batch.get("fixed_PTV")
            val_moving_PTV = batch.get("moving_PTV")
            os.makedirs(os.path.join(self.rootdir, path), exist_ok=True)

            ddf_samples = sample_ddf(ddf_mu, log_variance, dvf2ddf=self.dvf2ddf,
                                      correlation_length=self.correlation_length, samples=20)
            sample_contours(ddf_samples, val_frac_label, val_fixed_label, self.warp_layer,
                             self.num_class, path, self.rootdir, metadata, val_moving_PTV, val_fixed_PTV,
                             dice_metric=self.dice_metric)
            if val_frac_dose is not None:
                sample_warped_doses(ddf_samples, val_frac_dose, self.warp_layer, path, self.rootdir, metadata)

        values = {
            "validate/DSC_LUNG_LEFT": dice[0], "validate/DSC_LUNG_RIGHT": dice[1],
            "validate/DSC_SPINALKANAL": dice[2], "validate/DSC_TRACHEA": dice[3],
            "validate/DSC_AORTA": dice[4], "validate/DSC_ESOPHAGUS": dice[5],
            "validate/DSC_HEART": dice[6], "validate/DSC_GTV": dice[7],
            "validate/SSIM": ssim[0], "validate/folding_percentage": per_folding,
        }
        if tre_stats is not None:
            values.update(tre_stats)
        self.log_dict(values, on_epoch=True, on_step=True, batch_size=1)


def compute_jacobian_determinant(ddf):
    """Jacobian determinant det(I + grad(ddf)) of a DDF, [B, 3, H, W, D] -> [B, H, W, D]."""

    def gradient_3d(field, dim):
        grad = torch.zeros_like(field)
        if dim == 2:
            grad[:, :, 1:-1] = (field[:, :, 2:] - field[:, :, :-2]) / 2.0
            grad[:, :, 0] = field[:, :, 1] - field[:, :, 0]
            grad[:, :, -1] = field[:, :, -1] - field[:, :, -2]
        elif dim == 3:
            grad[:, :, :, 1:-1] = (field[:, :, :, 2:] - field[:, :, :, :-2]) / 2.0
            grad[:, :, :, 0] = field[:, :, :, 1] - field[:, :, :, 0]
            grad[:, :, :, -1] = field[:, :, :, -1] - field[:, :, :, -2]
        elif dim == 4:
            grad[:, :, :, :, 1:-1] = (field[:, :, :, :, 2:] - field[:, :, :, :, :-2]) / 2.0
            grad[:, :, :, :, 0] = field[:, :, :, :, 1] - field[:, :, :, :, 0]
            grad[:, :, :, :, -1] = field[:, :, :, :, -1] - field[:, :, :, :, -2]
        return grad

    dx, dy, dz = ddf[:, 0:1], ddf[:, 1:2], ddf[:, 2:3]

    dx_dh, dx_dw, dx_dd = gradient_3d(dx, 2)[:, 0], gradient_3d(dx, 3)[:, 0], gradient_3d(dx, 4)[:, 0]
    dy_dh, dy_dw, dy_dd = gradient_3d(dy, 2)[:, 0], gradient_3d(dy, 3)[:, 0], gradient_3d(dy, 4)[:, 0]
    dz_dh, dz_dw, dz_dd = gradient_3d(dz, 2)[:, 0], gradient_3d(dz, 3)[:, 0], gradient_3d(dz, 4)[:, 0]

    J00, J01, J02 = 1.0 + dx_dh, dx_dw, dx_dd
    J10, J11, J12 = dy_dh, 1.0 + dy_dw, dy_dd
    J20, J21, J22 = dz_dh, dz_dw, 1.0 + dz_dd

    det = (J00 * (J11 * J22 - J12 * J21)
           - J01 * (J10 * J22 - J12 * J20)
           + J02 * (J10 * J21 - J11 * J20))
    return det


def kl_divergence_frequency_domain(log_variance, correlation_length, prior_variance):
    """KL divergence between the learned (Gaussian, per-voxel) posterior and a Gaussian-process
    prior that shares the same frequency-domain correlation kernel (see paper appendix).

    Because prior and posterior share the correlation structure, the KL reduces to a per-voxel
    comparison of variances: KL = 0.5 * sum_i [ sigma_i^2/sigma_prior^2 - 1 - log(sigma_i^2/sigma_prior^2) ].
    """
    B, C, H, W, D = log_variance.shape
    var = torch.exp(log_variance)
    ratio = var / prior_variance
    kl_per_voxel = 0.5 * (ratio - 1 - torch.log(ratio + 1e-8))
    kl = kl_per_voxel.sum(dim=[1, 2, 3, 4])
    return kl.mean() / (C * H * W * D)


def frequency_domain_sampling(log_variance, correlation_length=8.0, spacing=(1.5, 1.5, 3.0), outputeps=False):
    """Sample reparameterized noise with anisotropic spatial correlation via frequency-domain filtering.

    Args:
        log_variance: predicted per-voxel log-variance, [B, C, H, W, D]
        correlation_length: spatial correlation length in mm
        spacing: physical voxel spacing (sx, sy, sz) in mm
    """
    B, C, H, W, D = log_variance.shape
    device, dtype = log_variance.device, log_variance.dtype

    white_noise = torch.randn(B, C, H, W, D, device=device, dtype=dtype)
    freq_noise = fft.fftn(white_noise, dim=(-3, -2, -1))

    freq_h = fft.fftfreq(H, device=device, dtype=dtype).view(-1, 1, 1) / spacing[0]
    freq_w = fft.fftfreq(W, device=device, dtype=dtype).view(1, -1, 1) / spacing[1]
    freq_d = fft.fftfreq(D, device=device, dtype=dtype).view(1, 1, -1) / spacing[2]

    sigma_freq = 1.0 / (2 * torch.pi * correlation_length)
    gaussian_filter = torch.exp(-0.5 * (freq_h ** 2 + freq_w ** 2 + freq_d ** 2) / sigma_freq ** 2)
    gaussian_filter = gaussian_filter.unsqueeze(0).unsqueeze(0)

    filtered_freq = freq_noise * gaussian_filter
    correlated_noise = fft.ifftn(filtered_freq, dim=(-3, -2, -1)).real
    correlated_noise = correlated_noise - correlated_noise.mean(dim=(-3, -2, -1), keepdim=True)
    correlated_noise = correlated_noise / (correlated_noise.std(dim=(-3, -2, -1), keepdim=True) + 1e-8)

    # Detach so the correlation pattern does not leak gradient into dvf_mu;
    # gradients still flow through std (log_variance).
    std = torch.exp(log_variance / 2.0)
    noise = std * correlated_noise.detach()
    if outputeps:
        return noise, correlated_noise
    return noise


def sample_ddf(ddf_mu, log_variance, dvf2ddf, correlation_length, samples: int = 20):
    """Draw multiple DDF samples from the learned distribution, [S, B, 3, H, W, D]."""
    ddf_samples = [ddf_mu + frequency_domain_sampling(log_variance, correlation_length) for _ in range(samples)]
    return torch.stack(ddf_samples, dim=0)


def sample_contours(ddf_samples, val_frac_label, val_fixed_label, warper, num_class, path, rootdir, metadata,
                     val_moving_PTV=None, val_fixed_PTV=None, dice_metric=None):
    """Warp each contour with every DDF sample, average to a mean probability map, and save
    mean/ground-truth contours (.npy + .mha) for downstream reliability-diagram analysis."""
    save_dir = os.path.join(rootdir, path)
    os.makedirs(save_dir, exist_ok=True)

    val_frac_label = one_hot(val_frac_label, num_classes=num_class)
    val_fixed_label_one_hot = one_hot(val_fixed_label, num_classes=num_class)
    organ_names = ['blank', 'Lung_Left', 'Lung_Right', 'Spinal_canal', 'Trachea', 'Aorta', 'Esophagus', 'Heart', 'GTV']

    include_ptv = val_moving_PTV is not None and val_fixed_PTV is not None
    if include_ptv:
        val_frac_label = torch.cat([val_frac_label, val_moving_PTV], dim=1)
        val_fixed_label_one_hot = torch.cat([val_fixed_label_one_hot, val_fixed_PTV], dim=1)
        organ_names.append('PTV')

    mean_contours = []
    gtv_dsc_scores = []
    for cls_idx in range(len(organ_names)):
        organ_mask = val_frac_label[:, cls_idx:cls_idx + 1]
        organ_samples = []
        for i in range(ddf_samples.shape[0]):
            pred = warper(organ_mask, ddf_samples[i])
            if cls_idx == 8:  # GTV: track the best-DSC sample for downstream inspection
                pred_binary = (pred > 0.5).float()
                dsc = dice_metric(y_pred=pred_binary, y=val_fixed_label_one_hot[:, cls_idx:cls_idx + 1])
                dsc_val = float(dsc[0][0].detach().cpu())
                if not np.isnan(dsc_val):
                    gtv_dsc_scores.append((i, dsc_val))
            organ_samples.append(pred)
        organ_mean = torch.mean(torch.cat(organ_samples, dim=0), dim=0, keepdim=True)
        mean_contours.append(organ_mean[0, 0].detach().cpu().numpy())

        organ_arr = organ_mean[0, 0].detach().cpu().permute(2, 1, 0).numpy()
        savemha(1 - organ_arr if cls_idx == 0 else organ_arr, f"mean_{organ_names[cls_idx]}", metadata, rootdir)
        savemha(val_fixed_label_one_hot[0, cls_idx].detach().cpu().permute(2, 1, 0).numpy(),
                f"gt_{organ_names[cls_idx]}", metadata, rootdir)

    np.save(os.path.join(save_dir, "mean_contours.npy"), np.stack(mean_contours))
    np.save(os.path.join(save_dir, "gt_labels.npy"), val_fixed_label_one_hot[0].detach().cpu().numpy())

    best_sample_idx = max(gtv_dsc_scores, key=lambda x: x[1])[0] if gtv_dsc_scores else 0
    return best_sample_idx


def sample_warped_doses(ddf_samples, val_frac_dose, warper, path, rootdir, metadata):
    """Warp a dose distribution with every DDF sample; save the stacked samples plus mean/std maps."""
    save_dir = os.path.join(rootdir, path)
    os.makedirs(save_dir, exist_ok=True)

    dose_samples = []
    for i in range(ddf_samples.shape[0]):
        pred_dose = warper(val_frac_dose, ddf_samples[i])
        dose_samples.append(pred_dose[0][0].detach().cpu().permute(2, 1, 0).numpy())

    dose_samples_np = np.stack(dose_samples)
    np.save(os.path.join(save_dir, "dose_samples.npy"), dose_samples_np)
    savemha(np.mean(dose_samples_np, axis=0), "dose_mean_sampled", metadata, rootdir)
    savemha(np.std(dose_samples_np, axis=0), "dose_std_sampled", metadata, rootdir)
    return dose_samples_np


def _trilinear_interpolate(volume, point):
    """Trilinear interpolation of `volume` ((C,)H,W,D) at fractional voxel coordinate `point`."""
    has_channels = volume.ndim == 4
    x, y, z = float(point[0]), float(point[1]), float(point[2])
    x0, y0, z0 = int(np.floor(x)), int(np.floor(y)), int(np.floor(z))
    x1, y1, z1 = x0 + 1, y0 + 1, z0 + 1

    shape = volume.shape[-3:] if has_channels else volume.shape
    x0c, x1c = np.clip([x0, x1], 0, shape[0] - 1)
    y0c, y1c = np.clip([y0, y1], 0, shape[1] - 1)
    z0c, z1c = np.clip([z0, z1], 0, shape[2] - 1)
    xd, yd, zd = x - x0, y - y0, z - z0

    if has_channels:
        c00 = volume[:, x0c, y0c, z0c] * (1 - xd) + volume[:, x1c, y0c, z0c] * xd
        c01 = volume[:, x0c, y0c, z1c] * (1 - xd) + volume[:, x1c, y0c, z1c] * xd
        c10 = volume[:, x0c, y1c, z0c] * (1 - xd) + volume[:, x1c, y1c, z0c] * xd
        c11 = volume[:, x0c, y1c, z1c] * (1 - xd) + volume[:, x1c, y1c, z1c] * xd
    else:
        c00 = volume[x0c, y0c, z0c] * (1 - xd) + volume[x1c, y0c, z0c] * xd
        c01 = volume[x0c, y0c, z1c] * (1 - xd) + volume[x1c, y0c, z1c] * xd
        c10 = volume[x0c, y1c, z0c] * (1 - xd) + volume[x1c, y1c, z0c] * xd
        c11 = volume[x0c, y1c, z1c] * (1 - xd) + volume[x1c, y1c, z1c] * xd

    c0 = c00 * (1 - yd) + c10 * yd
    c1 = c01 * (1 - yd) + c11 * yd
    return c0 * (1 - zd) + c1 * zd


def transform_landmarks(fix, mov, ddf_array, sigma_ddf_array, savedir, savepath, spacing=(1.5, 1.5, 3), save=False):
    """Compute target registration error (TRE) and predicted uncertainty at each landmark pair.

    `fix`/`mov` are [3, N] voxel coordinates (fractional allowed). Returns pre-registration
    landmark distances, predicted sigma at each landmark, and post-registration TRE (all mm).
    """
    tres, oridiff, sigmas = [], [], []
    for i in range(np.shape(fix)[1]):
        oridiff.append(np.linalg.norm((fix[:, i] - mov[:, i]) * spacing))
        deformation = _trilinear_interpolate(ddf_array, (fix[0, i], fix[1, i], fix[2, i]))
        tres.append(np.linalg.norm((fix[:, i] + deformation - mov[:, i]) * spacing))
        sigmas.append(_trilinear_interpolate(sigma_ddf_array, (fix[0, i], fix[1, i], fix[2, i])))

        if save:
            posdeform = (fix[:, i] + deformation) * spacing
            with open(f"{savedir}/{savepath.replace('.mha', '')}_landmark_values_mov.txt", "a") as f:
                f.write(f"{mov[0, i] * spacing[0]} {mov[1, i] * spacing[1]} {mov[2, i] * spacing[2]} 0 0\n")
            with open(f"{savedir}/{savepath.replace('.mha', '')}_landmark_values_fix.txt", "a") as f:
                f.write(f"{fix[0, i] * spacing[0]} {fix[1, i] * spacing[1]} {fix[2, i] * spacing[2]} 0 0\n")
            with open(f"{savedir}/{savepath.replace('.mha', '')}_landmark_values_pred.txt", "a") as f:
                f.write(f"{posdeform[0]} {posdeform[1]} {posdeform[2]} 0 0\n")

    return np.asarray(oridiff), np.asarray(sigmas), np.asarray(tres)


def createTREimage(mov, fix, val_fixed_image, ddf_array, radius=5, spacing=(1.5, 1.5, 3)):
    """Render per-landmark TRE as small disks on a blank volume, for visual QA overlays."""
    img_shape = val_fixed_image[0][0].shape
    img = np.zeros(img_shape, dtype=np.float32)
    for i in range(np.shape(fix)[1]):
        deformation = _trilinear_interpolate(ddf_array, (fix[0, i], fix[1, i], fix[2, i]))
        value = np.linalg.norm((fix[:, i] + deformation - mov[:, i]) * spacing)

        posdeform_int = np.round(mov[:, i]).astype(int)
        z = posdeform_int[2]
        if 0 <= z < img_shape[2]:
            img_slice = img[:, :, z]
            y0, x0 = int(posdeform_int[1]), int(posdeform_int[0])
            h, w = img_slice.shape
            for y in range(h):
                for x in range(w):
                    if (x - x0) ** 2 + (y - y0) ** 2 <= radius ** 2:
                        img_slice[x, y] = float(value)
            img[:, :, z] = img_slice
    return img


def savemha(array, filename, metadata, savedir):
    """Write a numpy array as .mha, copying spacing/origin/direction from `metadata`."""
    img = sitk.GetImageFromArray(array)
    img.SetDirection(metadata['direction'].cpu().numpy()[0])
    img.SetSpacing(metadata['spacing'].cpu().numpy()[0])
    img.SetOrigin(metadata['origin'].cpu().numpy()[0])
    path = metadata['path'][0].replace('/', '_').replace('.mha', '_')[1:].replace('original_', '')
    os.makedirs(os.path.join(savedir, path), exist_ok=True)
    sitk.WriteImage(img, os.path.join(savedir, path, filename + ".mha"))
