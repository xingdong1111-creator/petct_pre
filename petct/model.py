import hashlib

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv3d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Down3d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool3d(2), DoubleConv3d(in_ch, out_ch))

    def forward(self, x):
        return self.net(x)


class UNet3dEncoder(nn.Module):
    """只保留 U-Net 的 encoder，用于把 PET/CT 体积变成 embedding。"""

    def __init__(self, in_ch=2, base=32, out_dim=256):
        super().__init__()
        self.inc = DoubleConv3d(in_ch, base)
        self.d1 = Down3d(base, base * 2)
        self.d2 = Down3d(base * 2, base * 4)
        self.d3 = Down3d(base * 4, base * 8)
        self.out_dim = out_dim

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.d1(x1)
        x3 = self.d2(x2)
        x4 = self.d3(x3)
        # x4: (B, base*8, D/8, H/8, W/8)
        z = F.adaptive_avg_pool3d(x4, 1).flatten(1)
        return {"bottleneck": x4, "embedding": z}


class TabularEncoder(nn.Module):
    def __init__(self, in_dim, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, out_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class HashingTextEncoder(nn.Module):
    """轻量文本编码器：把基因突变文本 hash 成词袋，再做 embedding 平均。

    这是一个不依赖外部模型的 baseline。后续可替换为 Qwen/LLM encoder。
    """

    def __init__(self, vocab_size=4096, out_dim=64):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, out_dim)
        self.proj = nn.Sequential(nn.Linear(out_dim, out_dim), nn.ReLU())

    def _token_id(self, token):
        h = hashlib.md5(token.encode("utf-8")).hexdigest()
        return int(h[:8], 16) % self.vocab_size

    def forward(self, texts):
        # texts: List[str]，每个元素是一个患者的基因突变描述
        embs = []
        for text in texts:
            tokens = text.replace("；", " ").replace(";", " ").replace(",", " ").split()
            if not tokens:
                tokens = ["<empty>"]
            ids = torch.tensor([self._token_id(t) for t in tokens], dtype=torch.long)
            embs.append(self.embedding(ids).mean(dim=0))
        x = torch.stack(embs, dim=0)
        return self.proj(x)


class DeepHitModel(nn.Module):
    def __init__(self, cfg, clinical_in_dim):
        super().__init__()
        self.cfg = cfg
        self.image_encoder = UNet3dEncoder(cfg.in_channels, cfg.base_channels, cfg.image_dim)
        self.tab_encoder = TabularEncoder(clinical_in_dim, cfg.clinical_dim)
        self.text_encoder = HashingTextEncoder(cfg.text_vocab_size, cfg.text_dim)
        self.fusion = nn.Sequential(
            nn.Linear(cfg.image_dim + cfg.clinical_dim + cfg.text_dim, cfg.fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(cfg.fusion_dim, cfg.fusion_dim),
            nn.ReLU(),
        )
        self.head = nn.Linear(cfg.fusion_dim, cfg.time_bins)

    def forward(self, volume, clinical, genetic_text):
        img = self.image_encoder(volume)["embedding"]
        tab = self.tab_encoder(clinical)
        txt = self.text_encoder(genetic_text)
        z = self.fusion(torch.cat([img, tab, txt], dim=1))
        return self.head(z)  # 每个月的 hazard logit

    @torch.no_grad()
    def predict(self, volume, clinical, genetic_text):
        logits = self.forward(volume, clinical, genetic_text)
        from .loss import hazard_to_pmf

        p = hazard_to_pmf(logits)
        months = torch.arange(1, self.cfg.time_bins + 1, device=p.device, dtype=p.dtype)
        return {
            "prob": p,
            "risk_12m": p[:, :12].sum(dim=1),
            "risk_24m": p[:, :24].sum(dim=1),
            "expected_months": (p * months).sum(dim=1),
            "most_likely_month": p.argmax(dim=1) + 1,
        }
