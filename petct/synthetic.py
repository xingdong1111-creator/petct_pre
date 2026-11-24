import json
from pathlib import Path

import numpy as np
import pandas as pd


def _add_sphere(volume, center, radius=4, value=5.0):
    d, h, w = np.indices(volume.shape)
    mask = (d - center[0]) ** 2 + (h - center[1]) ** 2 + (w - center[2]) ** 2 <= radius**2
    volume[mask] += value


def make_synthetic_dataset(out_dir, n=64, seed=0):
    """生成一个可学习的小型合成数据集，用于 smoke test。"""
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    vol_dir = out_dir / "volumes"
    vol_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for i in range(n):
        pid = f"p{i:04d}"
        d = int(rng.integers(300, 601))
        h = w = 64
        ct = rng.normal(0, 0.3, size=(d, h, w)).astype(np.float32)
        pet = rng.normal(0, 0.3, size=(d, h, w)).astype(np.float32)

        # 合成“病灶信号”：越亮，风险越高
        strength = float(rng.uniform(0.2, 1.0))
        center = (int(rng.integers(20, d - 20)), 32, 32)
        _add_sphere(pet, center, radius=4, value=4 * strength)
        _add_sphere(ct, center, radius=4, value=2 * strength)

        age = float(rng.uniform(40, 85))
        sex = int(rng.integers(0, 2))
        smoking = int(rng.integers(0, 2))
        stage = int(rng.integers(1, 5))
        suvmax = float(2 + 10 * strength + rng.normal(0, 0.5))
        mtv = float(5 + 50 * strength + rng.normal(0, 2))
        tlg = suvmax * mtv / 10.0

        risk = 0.5 * strength + 0.2 * (age - 40) / 45 + 0.2 * (suvmax / 12) + 0.1 * (stage / 4)
        event = int(rng.random() < 0.65)
        if event:
            time_months = float(np.clip(60 - 45 * risk + rng.normal(0, 3), 1, 60))
        else:
            time_months = float(rng.uniform(24, 60))

        genetic_text = rng.choice(
            [
                "EGFR 19del 阳性",
                "EGFR L858R 阳性",
                "ALK 融合阴性",
                "KRAS G12C 阳性，TP53 突变",
                "未检出明确驱动基因突变",
            ]
        )

        ct_path = vol_dir / f"{pid}_ct.npy"
        pet_path = vol_dir / f"{pid}_pet.npy"
        np.save(ct_path, ct)
        np.save(pet_path, pet)

        rows.append(
            {
                "patient_id": pid,
                "ct_path": str(ct_path),
                "pet_path": str(pet_path),
                "time_months": time_months,
                "event": event,
                "genetic_text": genetic_text,
                "clinical_json": json.dumps(
                    {
                        "age": age,
                        "sex": sex,
                        "smoking": smoking,
                        "stage": stage,
                        "suvmax": suvmax,
                        "mtv": mtv,
                        "tlg": tlg,
                    }
                ),
            }
        )

    manifest = out_dir / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    return manifest
