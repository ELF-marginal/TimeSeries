import argparse
import json
import os
import random
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from data_provider.data_loader_ts import TSForecastDataset
from model.iTransformer import Model as ITransformer


def parse_pred_lens(text):
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model_config(args, pred_len):
    return SimpleNamespace(
        seq_len=args.seq_len,
        label_len=args.label_len,
        pred_len=pred_len,
        enc_in=args.enc_in,
        dec_in=args.enc_in,
        c_out=args.enc_in,
        d_model=args.d_model,
        n_heads=args.n_heads,
        e_layers=args.e_layers,
        d_layers=1,
        d_ff=args.d_ff,
        moving_avg=25,
        factor=args.factor,
        distil=True,
        dropout=args.dropout,
        embed="fixed",
        freq="h",
        activation=args.activation,
        output_attention=False,
        class_strategy="projection",
        use_norm=args.use_norm,
    )


def forward_batch(model, batch_x, batch_y, args, pred_len, device, use_amp=False):
    batch_x = batch_x.float().to(device)
    batch_y = batch_y.float().to(device)

    dec_zeros = torch.zeros_like(batch_y[:, -pred_len:, :])
    dec_inp = torch.cat([batch_y[:, : args.label_len, :], dec_zeros], dim=1).float().to(device)

    if use_amp:
        with torch.cuda.amp.autocast():
            outputs = model(batch_x, None, dec_inp, None)
    else:
        outputs = model(batch_x, None, dec_inp, None)

    return outputs[:, -pred_len:, :], batch_y[:, -pred_len:, :]


def evaluate(model, loader, criterion, args, pred_len, device):
    model.eval()
    losses = []
    with torch.no_grad():
        for batch_x, batch_y, _, _ in loader:
            outputs, target = forward_batch(model, batch_x, batch_y, args, pred_len, device)
            losses.append(criterion(outputs, target).item())
    model.train()
    return float(np.mean(losses))


def train_one_horizon(args, pred_len, device):
    train_set = TSForecastDataset(
        args.train_csv,
        flag="train",
        seq_len=args.seq_len,
        label_len=args.label_len,
        pred_len=pred_len,
        train_ratio=args.train_ratio,
    )
    val_set = TSForecastDataset(
        args.train_csv,
        flag="val",
        seq_len=args.seq_len,
        label_len=args.label_len,
        pred_len=pred_len,
        train_ratio=args.train_ratio,
        standardizer=train_set.standardizer,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=device.type == "cuda",
    )

    config = build_model_config(args, pred_len)
    model = ITransformer(config).float().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp and device.type == "cuda")

    run_dir = os.path.join(args.checkpoints, f"itransformer_pl{pred_len}")
    os.makedirs(run_dir, exist_ok=True)
    train_set.standardizer.save(os.path.join(run_dir, "scaler.npz"))

    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(config), f, indent=2)

    best_val = float("inf")
    bad_epochs = 0
    history = []

    print(f"\n===== Training pred_len={pred_len} =====")
    for epoch in range(1, args.train_epochs + 1):
        model.train()
        train_losses = []
        for batch_x, batch_y, _, _ in train_loader:
            optimizer.zero_grad(set_to_none=True)
            if args.use_amp and device.type == "cuda":
                outputs, target = forward_batch(model, batch_x, batch_y, args, pred_len, device, use_amp=True)
                loss = criterion(outputs, target)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs, target = forward_batch(model, batch_x, batch_y, args, pred_len, device)
                loss = criterion(outputs, target)
                loss.backward()
                optimizer.step()
            train_losses.append(loss.item())

        train_loss = float(np.mean(train_losses))
        val_loss = evaluate(model, val_loader, criterion, args, pred_len, device)
        history.append({"epoch": epoch, "train_mse": train_loss, "val_mse": val_loss})
        print(f"Epoch {epoch:03d} | train_mse={train_loss:.6f} | val_mse={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            bad_epochs = 0
            torch.save(model.state_dict(), os.path.join(run_dir, "checkpoint.pth"))
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping at epoch {epoch}; best val_mse={best_val:.6f}")
                break

    with open(os.path.join(run_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump({"best_val_mse": best_val, "history": history}, f, indent=2)

    return best_val


def main():
    parser = argparse.ArgumentParser(description="Train iTransformer for the TS100 forecasting task.")
    parser.add_argument("--train_csv", default="train/train.csv")
    parser.add_argument("--checkpoints", default="checkpoints")
    parser.add_argument("--pred_lens", default="96,192,336,720")
    parser.add_argument("--seq_len", type=int, default=96)
    parser.add_argument("--label_len", type=int, default=48)
    parser.add_argument("--enc_in", type=int, default=100)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--train_epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--n_heads", type=int, default=12)
    parser.add_argument("--e_layers", type=int, default=3)
    parser.add_argument("--d_ff", type=int, default=3072)
    parser.add_argument("--factor", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--activation", default="gelu")
    parser.add_argument("--use_norm", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--use_amp", action="store_true", default=True)
    parser.add_argument("--no_amp", action="store_false", dest="use_amp")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    args.train_csv = os.path.join(base_dir, args.train_csv)
    args.checkpoints = os.path.join(base_dir, args.checkpoints)

    set_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    results = {}
    for pred_len in parse_pred_lens(args.pred_lens):
        results[str(pred_len)] = train_one_horizon(args, pred_len, device)

    avg = float(np.mean(list(results.values())))
    print("\nValidation summary:")
    for pred_len, mse in results.items():
        print(f"pred_len={pred_len}: val_mse={mse:.6f}")
    print(f"average_val_mse={avg:.6f}")


if __name__ == "__main__":
    main()
