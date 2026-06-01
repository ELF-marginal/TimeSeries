import argparse
import json
import os
from types import SimpleNamespace

import numpy as np
import torch

from data_provider.data_loader_ts import Standardizer
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


def predict_horizon(args, pred_len, history, device):
    run_dir = os.path.join(args.checkpoints, f"itransformer_pl{pred_len}")
    model, config = load_model(run_dir, device)
    standardizer = Standardizer.load(os.path.join(run_dir, "scaler.npz"))

    norm_history = standardizer.transform(history).astype(np.float32)
    preds = []

    with torch.no_grad():
        for start in range(0, len(norm_history), args.batch_size):
            batch = torch.from_numpy(norm_history[start:start + args.batch_size]).float().to(device)
            batch_size = batch.shape[0]
            dec_inp = torch.zeros(batch_size, config.label_len + pred_len, config.enc_in, device=device)
            out = model(batch, None, dec_inp, None)
            preds.append(out[:, -pred_len:, :].detach().cpu().numpy())

    pred = np.concatenate(preds, axis=0)
    pred = standardizer.inverse_transform(pred).astype(np.float32)

    expected_shape = (history.shape[0], pred_len, history.shape[2])
    if pred.shape != expected_shape:
        raise ValueError(f"pred_{pred_len}.npy shape mismatch: expected {expected_shape}, got {pred.shape}")

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"pred_{pred_len}.npy")
    np.save(output_path, pred)
    print(f"Saved {output_path}: {pred.shape}")


def main():
    parser = argparse.ArgumentParser(description="Generate TS100 submission predictions with trained iTransformer models.")
    parser.add_argument("--test_npy", default="test_demo/hist_96.npy")
    parser.add_argument("--checkpoints", default="checkpoints")
    parser.add_argument("--output_dir", default="predictions")
    parser.add_argument("--pred_lens", default="96,192,336,720")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    args.test_npy = os.path.join(base_dir, args.test_npy)
    args.checkpoints = os.path.join(base_dir, args.checkpoints)
    args.output_dir = os.path.join(base_dir, args.output_dir)

    history = np.load(args.test_npy).astype(np.float32)
    if history.ndim != 3 or history.shape[1:] != (96, 100):
        raise ValueError(f"Expected test input shape (N, 96, 100), got {history.shape}")

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    for pred_len in parse_pred_lens(args.pred_lens):
        predict_horizon(args, pred_len, history, device)


if __name__ == "__main__":
    main()
