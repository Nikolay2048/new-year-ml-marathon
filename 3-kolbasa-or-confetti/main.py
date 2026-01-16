import os
import math
import time
import zipfile
import random
import argparse
from dataclasses import dataclass
from typing import Tuple, Dict, Any, List

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights


# -------------------------
# Utils
# -------------------------
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def unzip_if_needed(zip_path: str, out_dir: str):
    ensure_dir(out_dir)
    marker = os.path.join(out_dir, ".unzipped_ok")
    if os.path.exists(marker):
        return
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)
    with open(marker, "w") as f:
        f.write("ok")


# -------------------------
# Metric (пример)
# -------------------------
@torch.no_grad()
def mae(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    return (y_pred - y_true).abs().mean()


# -------------------------
# Dataset
# -------------------------
class CoversDataset(Dataset):
    def __init__(self, image_dir: str, df: pd.DataFrame = None, transform=None, is_test: bool = False):
        """
        df: columns ['image_id', 'c', 's'] for train/val
        if is_test=True, df can contain only ['image_id']
        image_dir contains *.jpg with names like img_000001.jpg / test_000001.jpg
        """
        self.image_dir = image_dir
        self.df = df.reset_index(drop=True) if df is not None else None
        self.transform = transform
        self.is_test = is_test

        if self.df is None:
            raise ValueError("df is required")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image_id = row["image_id"]
        img_path = os.path.join(self.image_dir, f"{image_id}.jpg")
        img = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        if self.is_test:
            return image_id, img

        target = torch.tensor([row["c"], row["s"]], dtype=torch.float32)
        return image_id, img, target


# -------------------------
# Model
# -------------------------
class RegressionResNet18(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.backbone = None

        if pretrained:
            try:
                self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
            except Exception:
                # если нет интернета / веса не скачались
                self.backbone = resnet18(weights=None)
        else:
            self.backbone = resnet18(weights=None)

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 2),
        )

    def forward(self, x):
        feats = self.backbone(x)
        out = self.head(feats)
        # ограничим в [0,1] через sigmoid (сабмит всё равно клиппится)
        out = torch.sigmoid(out)
        return out


# -------------------------
# Train / Eval
# -------------------------
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    loss_name: str = "mse",
    max_grad_norm: float = 1.0,
):
    model.train()
    running_loss = 0.0
    running_mae = 0.0
    n = 0

    for _, x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            pred = model(x)
            if loss_name == "mse":
                loss = F.mse_loss(pred, y)
            elif loss_name == "smoothl1":
                loss = F.smooth_l1_loss(pred, y, beta=0.05)
            else:
                raise ValueError("loss_name must be mse or smoothl1")

        scaler.scale(loss).backward()

        if max_grad_norm is not None and max_grad_norm > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        scaler.step(optimizer)
        scaler.update()

        bs = x.size(0)
        running_loss += loss.item() * bs
        running_mae += mae(pred.detach(), y).item() * bs
        n += bs

    return running_loss / n, running_mae / n


@torch.no_grad()
def validate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    running_loss = 0.0
    running_mae = 0.0
    n = 0

    for _, x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        pred = model(x)
        loss = F.mse_loss(pred, y)

        bs = x.size(0)
        running_loss += loss.item() * bs
        running_mae += mae(pred, y).item() * bs
        n += bs

    return running_loss / n, running_mae / n


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> pd.DataFrame:
    model.eval()
    ids = []
    preds = []

    for image_id, x in loader:
        x = x.to(device, non_blocking=True)
        out = model(x).detach().cpu().numpy()
        ids.extend(list(image_id))
        preds.append(out)

    preds = np.concatenate(preds, axis=0)
    preds = np.clip(preds, 0.0, 1.0)
    return pd.DataFrame({"image_id": ids, "c": preds[:, 0], "s": preds[:, 1]})


# -------------------------
# Transforms
# -------------------------
class AddGaussianNoise(nn.Module):
    def __init__(self, std: float = 0.02, p: float = 0.5):
        super().__init__()
        self.std = std
        self.p = p

    def forward(self, x: torch.Tensor):
        if random.random() > self.p:
            return x
        noise = torch.randn_like(x) * self.std
        return torch.clamp(x + noise, 0.0, 1.0)


def build_transforms(img_size: int = 224):
    train_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.02),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.2),
        transforms.ToTensor(),
        AddGaussianNoise(std=0.02, p=0.4),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    val_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    return train_tfms, val_tfms


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--train_zip", type=str, default="./data/train_images_covers.zip")
    parser.add_argument("--test_zip", type=str, default="./data/test_images_covers.zip")
    parser.add_argument("--train_csv", type=str, default="./data/train_labels_covers.csv")
    parser.add_argument("--sample_sub", type=str, default="./data/sample_submission_covers.csv")

    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pretrained", action="store_true", help="использовать pretrained веса")
    parser.add_argument("--loss", type=str, default="smoothl1", choices=["mse", "smoothl1"])
    parser.add_argument("--out_dir", type=str, default="./outputs")
    parser.add_argument("--submission_name", type=str, default="submission.csv")

    args = parser.parse_args()

    seed_everything(args.seed)
    ensure_dir(args.out_dir)

    # 1) распаковка
    unzip_if_needed(args.train_zip, os.path.join(args.data_dir, "train_unz"))
    unzip_if_needed(args.test_zip, os.path.join(args.data_dir, "test_unz"))

    train_img_dir = os.path.join(args.data_dir, "train_unz", "train", "images")
    test_img_dir = os.path.join(args.data_dir, "test_unz", "test", "images")

    if not os.path.isdir(train_img_dir):
        raise FileNotFoundError(f"train images not found: {train_img_dir}")
    if not os.path.isdir(test_img_dir):
        raise FileNotFoundError(f"test images not found: {test_img_dir}")

    # 2) данные
    df = pd.read_csv(args.train_csv)
    df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    n_val = int(len(df) * args.val_ratio)
    df_val = df.iloc[:n_val].reset_index(drop=True)
    df_train = df.iloc[n_val:].reset_index(drop=True)

    train_tfms, val_tfms = build_transforms(args.img_size)

    ds_train = CoversDataset(train_img_dir, df_train, transform=train_tfms, is_test=False)
    ds_val = CoversDataset(train_img_dir, df_val, transform=val_tfms, is_test=False)

    dl_train = DataLoader(
        ds_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # 3) модель
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RegressionResNet18(pretrained=args.pretrained).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # простой scheduler (cosine)
    total_steps = args.epochs * len(dl_train)
    def lr_lambda(step):
        # cosine decay from 1 -> 0
        if total_steps <= 1:
            return 1.0
        t = step / (total_steps - 1)
        return 0.5 * (1.0 + math.cos(math.pi * t))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    best_val_mae = float("inf")
    best_path = os.path.join(args.out_dir, "best_model.pt")

    # 4) train loop
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        tr_loss, tr_mae = train_one_epoch(
            model=model,
            loader=dl_train,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            loss_name=args.loss,
            max_grad_norm=1.0,
        )

        # шаги scheduler по батчам: догоним по эпохе
        for _ in range(len(dl_train)):
            scheduler.step()
            global_step += 1

        va_loss, va_mae = validate(model=model, loader=dl_val, device=device)

        dt = time.time() - t0
        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train loss {tr_loss:.6f}, mae {tr_mae:.6f} | "
            f"val loss {va_loss:.6f}, mae {va_mae:.6f} | "
            f"time {dt:.1f}s"
        )

        if va_mae < best_val_mae:
            best_val_mae = va_mae
            torch.save({"model": model.state_dict(), "args": vars(args)}, best_path)
            print(f"  -> saved best to {best_path} (val_mae={best_val_mae:.6f})")

    # 5) инференс теста
    #    грузим лучший чекпойнт
    ckpt = torch.load(best_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    sample_sub = pd.read_csv(args.sample_sub)
    df_test = sample_sub[["image_id"]].copy()

    ds_test = CoversDataset(test_img_dir, df_test, transform=val_tfms, is_test=True)
    dl_test = DataLoader(
        ds_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    pred_df = predict(model=model, loader=dl_test, device=device)

    # гарантируем порядок как в sample_submission
    sub = sample_sub[["image_id"]].merge(pred_df, on="image_id", how="left")
    sub["c"] = sub["c"].fillna(0.0).clip(0.0, 1.0)
    sub["s"] = sub["s"].fillna(0.0).clip(0.0, 1.0)

    out_path = os.path.join(args.out_dir, args.submission_name)
    sub.to_csv(out_path, index=False)
    print(f"\nSaved submission to: {out_path}")
    print(sub.head())


if __name__ == "__main__":
    main()
