import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class Standardizer:
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def fit(self, data):
        self.mean = data.mean(axis=0, keepdims=True).astype(np.float32)
        self.std = data.std(axis=0, keepdims=True).astype(np.float32)
        self.std[self.std < 1e-6] = 1.0
        return self

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return data * self.std + self.mean

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path, mean=self.mean, std=self.std)

    @classmethod
    def load(cls, path):
        obj = np.load(path)
        return cls(mean=obj["mean"].astype(np.float32), std=obj["std"].astype(np.float32))


def read_train_csv(path):
    data = pd.read_csv(path).values.astype(np.float32)
    if data.ndim != 2:
        raise ValueError(f"Expected 2-D csv data, got shape {data.shape}")
    if data.shape[1] != 100:
        raise ValueError(f"Expected 100 feature columns, got {data.shape[1]}")
    return data


class TSForecastDataset(Dataset):
    def __init__(
        self,
        csv_path,
        flag,
        seq_len=96,
        label_len=48,
        pred_len=96,
        train_ratio=0.8,
        standardizer=None,
    ):
        if flag not in {"train", "val"}:
            raise ValueError("flag must be 'train' or 'val'")

        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.flag = flag

        raw = read_train_csv(csv_path)
        train_end = int(len(raw) * train_ratio)
        if train_end <= seq_len + pred_len:
            raise ValueError("Training split is too short for the requested window lengths.")

        if standardizer is None:
            standardizer = Standardizer().fit(raw[:train_end])
        self.standardizer = standardizer
        data = standardizer.transform(raw).astype(np.float32)

        if flag == "train":
            self.data = data[:train_end]
        else:
            self.data = data[max(0, train_end - seq_len):]

        min_len = seq_len + pred_len
        if len(self.data) < min_len:
            raise ValueError(f"{flag} split is too short: need at least {min_len}, got {len(self.data)}")

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = s_end + self.pred_len

        seq_x = self.data[s_begin:s_end]
        seq_y = self.data[r_begin:r_end]

        return (
            torch.from_numpy(seq_x),
            torch.from_numpy(seq_y),
            torch.empty(0),
            torch.empty(0),
        )
