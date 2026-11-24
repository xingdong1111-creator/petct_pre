from dataclasses import dataclass


@dataclass
class Config:
    # 输入体积下采样到固定大小；真实数据建议 64x128x128，smoke test 用小尺寸
    volume_d: int = 64
    volume_h: int = 64
    volume_w: int = 64

    # 3D U-Net encoder
    in_channels: int = 2  # PET + CT
    base_channels: int = 32
    image_dim: int = 256

    # 表格与文本
    clinical_dim: int = 64
    text_dim: int = 64
    text_vocab_size: int = 4096

    # 融合与输出
    fusion_dim: int = 128
    time_bins: int = 60  # 60 个月

    # 训练
    lr: float = 1e-4
    weight_decay: float = 1e-4
    batch_size: int = 2
    epochs: int = 3
    device: str = "cpu"
