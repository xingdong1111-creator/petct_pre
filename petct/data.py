import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


DEFAULT_CLINICAL_FEATURES = ["age", "sex", "smoking", "stage", "suvmax", "mtv", "tlg"]


def downsample_volume(volume, out_shape):
    """volume: (C, D, H, W) numpy/torch，重采样到固定大小。"""
    x = torch.as_tensor(volume, dtype=torch.float32).unsqueeze(0)
    x = F.interpolate(x, size=out_shape, mode="trilinear", align_corners=False)
    return x.squeeze(0)


def parse_clinical(value, feature_names=DEFAULT_CLINICAL_FEATURES):
    """把临床信息解析成数值向量。

    支持：
    - JSON 字符串，如 {"age": 65, "sex": 1, "smoking": 0, "stage": 3, ...}
    - dict
    - 单独的标量列（由调用方传入）
    """
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            value = {}
    if not isinstance(value, dict):
        value = {}
    vec = []
    for name in feature_names:
        v = value.get(name, 0.0)
        try:
            vec.append(float(v))
        except (TypeError, ValueError):
            vec.append(0.0)
    return vec


class PetCtDataset(Dataset):
    """PET/CT + 临床 + 基因文本 + 骨转移时间数据集。

    manifest CSV 需要的列：
    - patient_id
    - ct_path: CT .npy 路径，形状 (D, H, W)
    - pet_path: PET .npy 路径，形状 (D, H, W)
    - time_months: 骨转移时间或删失时间（月）
    - event: 1=发生骨转移，0=删失
    - genetic_text: 基因突变文本
    - clinical_json（可选）：临床信息 JSON；若无则读取 feature_names 对应的列
    """

    def __init__(self, manifest, cfg, feature_names=DEFAULT_CLINICAL_FEATURES):
        self.df = pd.read_csv(manifest)
        self.cfg = cfg
        self.feature_names = feature_names
        self.out_shape = (cfg.volume_d, cfg.volume_h, cfg.volume_w)

    def __len__(self):
        return len(self.df)

    def _load_volume(self, ct_path, pet_path):
        ct = np.load(ct_path)
        pet = np.load(pet_path)
        if ct.shape != pet.shape:
            raise ValueError(f"CT/PET 形状不一致: {ct_path} vs {pet_path}")
        volume = np.stack([pet, ct], axis=0)  # (2, D, H, W)
        return downsample_volume(volume, self.out_shape)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        volume = self._load_volume(row["ct_path"], row["pet_path"])

        if "clinical_json" in self.df.columns:
            clinical_value = row["clinical_json"]
        else:
            clinical_value = {name: row.get(name, 0.0) for name in self.feature_names}
        clinical = torch.tensor(parse_clinical(clinical_value, self.feature_names), dtype=torch.float32)

        genetic_text = str(row.get("genetic_text", ""))
        time_months = torch.tensor(float(row["time_months"]), dtype=torch.float32)
        event = torch.tensor(float(row["event"]), dtype=torch.float32)
        return {
            "patient_id": row["patient_id"],
            "volume": volume,
            "clinical": clinical,
            "genetic_text": genetic_text,
            "time_months": time_months,
            "event": event,
        }


def collate_fn(batch):
    return {
        "patient_id": [b["patient_id"] for b in batch],
        "volume": torch.stack([b["volume"] for b in batch], dim=0),
        "clinical": torch.stack([b["clinical"] for b in batch], dim=0),
        "genetic_text": [b["genetic_text"] for b in batch],
        "time_months": torch.stack([b["time_months"] for b in batch], dim=0),
        "event": torch.stack([b["event"] for b in batch], dim=0),
    }
