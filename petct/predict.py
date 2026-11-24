import argparse

import torch
from torch.utils.data import DataLoader

from .config import Config
from .data import PetCtDataset, collate_fn, DEFAULT_CLINICAL_FEATURES
from .model import DeepHitModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location=args.device)
    cfg = Config(**ckpt["config"])
    model = DeepHitModel(cfg, clinical_in_dim=len(ckpt["clinical_features"]))
    model.load_state_dict(ckpt["model"])
    model.to(args.device).eval()

    ds = PetCtDataset(args.manifest, cfg, feature_names=ckpt["clinical_features"])
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_fn)

    for batch in loader:
        volume = batch["volume"].to(args.device)
        clinical = batch["clinical"].to(args.device)
        out = model.predict(volume, clinical, batch["genetic_text"])
        print(
            f"{batch['patient_id'][0]}: "
            f"risk_12m={out['risk_12m'].item():.3f}, "
            f"risk_24m={out['risk_24m'].item():.3f}, "
            f"expected_months={out['expected_months'].item():.1f}, "
            f"most_likely_month={out['most_likely_month'].item()}"
        )


if __name__ == "__main__":
    main()
