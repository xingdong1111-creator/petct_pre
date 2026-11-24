import argparse
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import Config
from .data import PetCtDataset, collate_fn, DEFAULT_CLINICAL_FEATURES
from .loss import logistic_hazard_loss, hazard_to_pmf
from .metrics import concordance_index
from .model import DeepHitModel
from .synthetic import make_synthetic_dataset


def evaluate(model, loader, device):
    model.eval()
    times, events, risks = [], [], []
    with torch.no_grad():
        for batch in loader:
            volume = batch["volume"].to(device)
            clinical = batch["clinical"].to(device)
            logits = model(volume, clinical, batch["genetic_text"])
            risk_12m = hazard_to_pmf(logits)[:, :12].sum(dim=1).cpu().numpy()
            times.append(batch["time_months"].numpy())
            events.append(batch["event"].numpy())
            risks.append(risk_12m)  # 12 个月风险越高，越倾向早期转移
    model.train()
    return concordance_index(np.concatenate(times), np.concatenate(events), np.concatenate(risks))


def train(manifest, cfg, out_dir="checkpoints"):
    device = torch.device(cfg.device)
    ds = PetCtDataset(manifest, cfg)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn)
    model = DeepHitModel(cfg, clinical_in_dim=len(DEFAULT_CLINICAL_FEATURES)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    model.train()

    for epoch in range(1, cfg.epochs + 1):
        losses = []
        for batch in loader:
            volume = batch["volume"].to(device)
            clinical = batch["clinical"].to(device)
            logits = model(volume, clinical, batch["genetic_text"])
            loss = logistic_hazard_loss(logits, batch["time_months"], batch["event"])
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        cidx = evaluate(model, loader, device)
        print(f"epoch {epoch}: loss={np.mean(losses):.4f}, c_index={cidx:.4f}")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ckpt = out / "model.pt"
    torch.save({"model": model.state_dict(), "config": cfg.__dict__, "clinical_features": DEFAULT_CLINICAL_FEATURES}, ckpt)
    print(f"saved: {ckpt}")
    return ckpt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, default=None)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default="checkpoints")
    args = parser.parse_args()

    cfg = Config()
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.device is not None:
        cfg.device = args.device

    if args.synthetic:
        tmp = Path(tempfile.mkdtemp(prefix="petct_synth_"))
        manifest = make_synthetic_dataset(tmp, n=64)
        print(f"synthetic manifest: {manifest}")
    else:
        if not args.manifest:
            raise ValueError("请提供 --manifest，或使用 --synthetic")
        manifest = args.manifest

    train(manifest, cfg, args.out_dir)


if __name__ == "__main__":
    main()
