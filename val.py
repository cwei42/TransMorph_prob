"""Validate/evaluate a trained TM-prob checkpoint.

Example:
    python val.py --val-manifest data/val.json --ckpt logs_tmprob/.../checkpoints/epoch=199.ckpt
"""
import argparse
import os
from datetime import datetime

import torch
import lightning as L
from monai.data import DataLoader
from lightning.pytorch.loggers import TensorBoardLogger

from dataset import RegistrationData
from model import TMProb


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--val-manifest", required=True, help="JSON manifest of validation pairs")
    p.add_argument("--ckpt", required=True, help="path to a training checkpoint (.ckpt)")
    p.add_argument("--logdir", default="./logs_tmprob_val")
    p.add_argument("--correlation-length", type=float, default=8.0)
    p.add_argument("--num-label-class", type=int, default=9)
    p.add_argument("--spatial-size", type=int, nargs=3, default=(320, 224, 160))
    p.add_argument("--num-workers", type=int, default=12)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--save-samples-for", type=int, nargs="*", default=[],
                    help="batch indices for which to save sampled contours/doses/DDFs to disk")
    return p.parse_args()


def main():
    args = parse_args()
    L.seed_everything(42, workers=True)
    torch.set_float32_matmul_precision("high")

    data = RegistrationData(val_manifest=args.val_manifest, spatial_size=tuple(args.spatial_size))
    _, val_set = data.loaddata()
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=args.num_workers,
                             pin_memory=True, drop_last=True)

    rootdir = os.path.join(args.logdir, datetime.now().strftime("%Y-%m-%d-%H-%M"))
    os.makedirs(rootdir, exist_ok=True)

    net = TMProb(correlation_length=args.correlation_length, num_label_class=args.num_label_class,
                 label_weight=torch.ones(args.num_label_class - 1), savedir=rootdir)
    net.savelist = args.save_samples_for

    trainer = L.Trainer(
        accelerator="gpu",
        devices=[args.gpu],
        default_root_dir=rootdir,
        logger=TensorBoardLogger(save_dir=rootdir),
        enable_model_summary=False,
        limit_val_batches=1.0,
    )
    trainer.validate(net, dataloaders=val_loader, ckpt_path=args.ckpt)


if __name__ == "__main__":
    main()
