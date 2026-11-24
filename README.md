# petct_pre：肺癌 PET/CT 骨转移时间预测基线

输入：PET/CT 体积（每个患者约 300–600 层，层数可不同）、临床病理学数据、基因突变文本。
输出：未来 **60 个月**内发生骨转移的概率分布，并可换算为 12/24 个月风险和预计发生时间。

这是一个可运行的 **baseline**，用于验证数据格式、模型结构和训练流程，不代表临床可用模型。

## 模型结构

```text
PET/CT (B, 2, D, H, W)
  → 下采样到固定大小
  → 3D U-Net encoder（不要 decoder）
  → 全局池化 → 影像 embedding (B, 256)

临床病理特征 → MLP → 表格 embedding (B, 64)
基因突变文本 → HashingTextEncoder → 文本 embedding (B, 64)

拼接 → 融合 MLP → DeepHit 生存头 → (B, 60)
```

- **影像分支**：`UNet3dEncoder`，输入通道为 PET + CT，输出 256 维 embedding。
- **文本分支**：轻量 hash 词袋编码器，便于先跑通；后续可替换为 Qwen/LLM。
- **输出头**：DeepHit，把 60 个月离散成 60 个 bin，输出每个月骨转移概率。
- **参数量**：约 3.5M（base=32 时），适合中小样本量起步。

## 数据格式

准备一份 `manifest.csv`：

| 列名 | 含义 |
|---|---|
| `patient_id` | 患者 ID |
| `ct_path` | CT `.npy` 路径，形状 `(D, H, W)` |
| `pet_path` | PET `.npy` 路径，形状 `(D, H, W)` |
| `time_months` | 骨转移时间或删失时间（月） |
| `event` | `1`=发生骨转移，`0`=删失 |
| `genetic_text` | 基因突变文本 |
| `clinical_json` | 临床信息 JSON，例如 `{"age":65,"sex":1,"smoking":0,"stage":3,"suvmax":12.3,"mtv":45.2,"tlg":55.6}` |

注意：

- CT 和 PET 必须已经配准，形状一致。
- `time_months` 必须是 **这次 PET/CT 到骨转移确诊/末次随访的月数**。
- 未发生骨转移的患者属于删失（`event=0`），不要当成负样本二分类处理。
- 死亡且未转移患者较多时，应考虑竞争风险模型。

## 快速 smoke test

```bash
pip install -r requirements.txt
python -m petct.train --synthetic --epochs 3 --batch_size 2 --device cpu
```

程序会生成 64 个合成患者，训练 3 个 epoch，打印 loss 和简化版 C-index，并保存 `checkpoints/model.pt`。

## 训练真实数据

```bash
python -m petct.train \
  --manifest data/manifest.csv \
  --epochs 20 \
  --batch_size 2 \
  --device cuda \
  --out_dir checkpoints
```

## 推理

```bash
python -m petct.predict \
  --checkpoint checkpoints/model.pt \
  --manifest data/manifest.csv \
  --device cpu
```

输出示例：

```text
p0001: risk_12m=0.421, risk_24m=0.763, expected_months=18.4, most_likely_month=14
```

## 后续可扩展方向

1. **分块处理全体积**：300–600 层不必直接下采样，可切成 patch 后聚合，保留更多细节。
2. **肺/骨双 ROI**：分别编码肺原发灶和骨 ROI，再与临床特征融合。
3. **文本编码器升级为 Qwen**：把 `HashingTextEncoder` 换成 Qwen embedding，或把 3D token 与临床文本一起输入 Qwen。
4. **竞争风险**：若死亡且未转移患者较多，使用 competing-risk DeepHit 或 Fine-Gray 思路。
5. **评估**：增加 time-dependent AUC、Brier score、校准曲线和决策曲线。

## 重要试验设计提醒

- “骨转移前最后一次 PET/CT”容易引入 lead-time bias；建议明确时间起点，或采用 baseline / landmark 设计。
- 基因突变缺失不要当成“未突变”，应加 missing mask。
- 样本量约 4000 时，不建议从零训练大型 3D CNN；优先使用小模型或预训练 encoder。
