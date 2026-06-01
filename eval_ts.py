import argparse
import json
import os
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from data_provider.data_loader_ts import Standardizer, TSForecastDataset
from model.iTransformer import Model as ITransformer


def parse_pred_lens(text):
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def load_model(run_dir, device):
    with open(os.path.join(run_dir, "config.json"), "r", encoding="utf-8") as f:
        config = SimpleNamespace(**json.load(f))

    model = ITransformer(config).float().to(device)
    checkpoint = torch.load(os.path.join(run_dir, "checkpoint.pth"), map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    return model, config


def evaluate_one(args, pred_len, device):
    run_dir = os.path.join(args.checkpoints, f"itransformer_pl{pred_len}")
    model, config = load_model(run_dir, device)
    standardizer = Standardizer.load(os.path.join(run_dir, "scaler.npz"))

    val_set = TSForecastDataset(
        args.train_csv,
        flag="val",
        seq_len=config.seq_len,
        label_len=config.label_len,
        pred_len=pred_len,
        train_ratio=args.train_ratio,
        standardizer=standardizer,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )

    losses = []
    with torch.no_grad():
        for batch_x, batch_y, _, _ in val_loader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            dec_inp = torch.zeros(
                batch_y.shape[0],
                config.label_len + pred_len,
                config.enc_in,
                device=device,
            )
            pred = model(batch_x, None, dec_inp, None)[:, -pred_len:, :]
            true = batch_y[:, -pred_len:, :]
            pred = pred.detach().cpu().numpy()
            true = true.detach().cpu().numpy()
            pred = standardizer.inverse_transform(pred)
            true = standardizer.inverse_transform(true)
            losses.append(float(np.mean(np.square(true - pred))))

    return float(np.mean(losses))


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained checkpoints on the validation split.")
    parser.add_argument("--train_csv", default="train/train.csv")
    parser.add_argument("--checkpoints", default="checkpoints")
    parser.add_argument("--pred_lens", default="96,192,336,720")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    args.train_csv = os.path.join(base_dir, args.train_csv)
    args.checkpoints = os.path.join(base_dir, args.checkpoints)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    results = {}
    for pred_len in parse_pred_lens(args.pred_lens):
        mse = evaluate_one(args, pred_len, device)
        results[pred_len] = mse
        print(f"MSE {pred_len}: {mse:.6f}")

    avg = float(np.mean(list(results.values())))
    print(f"MSE Avg: {avg:.6f}")

    if avg < 0.005:
        print("Validation level: below 0.005 bonus line")
    elif avg < 0.006:
        print("Validation level: below 0.006 bonus line")
    elif avg < 0.01:
        print("Validation level: below 0.01 basic score line")
    else:
        print("Validation level: above 0.01; tune model or training setup")


if __name__ == "__main__":
    main()
