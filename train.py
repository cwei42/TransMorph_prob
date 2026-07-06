"""Train TM-prob.

Example:
    python train.py --train-manifest data/train.json --val-manifest data/val.json \
        --logdir ./logs --epochs 200
"""
import argparse
import os
from datetime import datetime

import torch
import lightning as L
from monai.data import DataLoader
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from dataset import RegistrationData
from model import TMProb


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train-manifest", required=True, help="JSON manifest of training pairs")
    p.add_argument("--val-manifest", required=True, help="JSON manifest of validation pairs")
    p.add_argument("--logdir", default="./logs_tmprob")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--kl-warmup-epochs", type=int, default=70)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lMSE", type=float, default=10)
    p.add_argument("--lBE", type=float, default=1)
    p.add_argument("--ldice", type=float, default=0)
    p.add_argument("--kl", type=float, default=1e-2)
    p.add_argument("--prior-lambda", type=float, default=150.0)
    p.add_argument("--correlation-length", type=float, default=8.0)
    p.add_argument("--num-label-class", type=int, default=9, help="organ classes + background")
    p.add_argument("--spatial-size", type=int, nargs=3, default=(320, 224, 160))
    p.add_argument("--num-workers", type=int, default=12)
    p.add_argument("--gpu", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    L.seed_everything(42, workers=True)
    torch.set_float32_matmul_precision("high")

    data = RegistrationData(train_manifest=args.train_manifest, val_manifest=args.val_manifest,
                             spatial_size=tuple(args.spatial_size))
    train_set, _ = data.loaddata()
    data_val = RegistrationData(val_manifest=args.val_manifest, spatial_size=tuple(args.spatial_size))
    _, val_set = data_val.loaddata()

    rootdir = os.path.join(args.logdir, datetime.now().strftime("%Y-%m-%d-%H-%M"))
    os.makedirs(rootdir, exist_ok=True)

    net = TMProb(lr=args.lr, lMSE=args.lMSE, lBE=args.lBE, ldice=args.ldice, kl=args.kl,
                 prior_lambda=args.prior_lambda, correlation_length=args.correlation_length,
                 num_label_class=args.num_label_class, kl_warmup_epochs=args.kl_warmup_epochs,
                 label_weight=torch.ones(args.num_label_class - 1), savedir=rootdir)

    train_loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=args.num_workers, drop_last=True)

    trainer = L.Trainer(
        accelerator="gpu",
        devices=[args.gpu],
        max_epochs=args.epochs,
        default_root_dir=rootdir,
        logger=TensorBoardLogger(save_dir=rootdir),
        callbacks=ModelCheckpoint(filename="{epoch}", save_top_k=-1),
        num_sanity_val_steps=1,
    )
    trainer.fit(net, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    main()
