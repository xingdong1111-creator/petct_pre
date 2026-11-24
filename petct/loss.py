import torch


def logistic_hazard_loss(logits, time_months, event):
    """离散时间 logistic hazard loss。

    logits: (B, K)，每个月的 hazard logit，sigmoid 后表示“活到这个月时，本月发生骨转移的概率”。
    time_months: (B,)，事件发生或删失的月份（从 1 开始）。
    event: (B,)，1=发生骨转移，0=删失。
    """
    h = torch.sigmoid(logits).clamp(1e-6, 1 - 1e-6)
    B, K = logits.shape
    months = torch.arange(1, K + 1, device=logits.device).unsqueeze(0)
    time_months = time_months.to(logits.device).float()
    event = event.to(logits.device).float()

    t = time_months.clamp(1, K).long()
    before = months < t.unsqueeze(1)
    log_surv_before = (torch.log1p(-h) * before).sum(dim=1)
    log_h_t = torch.log(h).gather(1, (t - 1).unsqueeze(1)).squeeze(1)
    event_loss = -(log_surv_before + log_h_t)

    c = time_months.clamp(0, K).long()
    upto = months <= c.unsqueeze(1)
    log_surv_c = (torch.log1p(-h) * upto).sum(dim=1)
    censor_loss = -log_surv_c

    return (event * event_loss + (1 - event) * censor_loss).mean()


def hazard_to_pmf(logits):
    """把 hazard 转成每个时间点的概率质量 P(T=t)。"""
    h = torch.sigmoid(logits).clamp(1e-6, 1 - 1e-6)
    surv_prev = torch.cumprod(1 - h, dim=1)
    surv_prev = torch.cat([torch.ones_like(surv_prev[:, :1]), surv_prev[:, :-1]], dim=1)
    return h * surv_prev


def expected_months_from_hazard(logits):
    pmf = hazard_to_pmf(logits)
    months = torch.arange(1, logits.shape[1] + 1, device=pmf.device, dtype=pmf.dtype)
    return (pmf * months).sum(dim=1)
