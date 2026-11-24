import numpy as np


def concordance_index(time_months, event, risk):
    """简化版 C-index：只比较发生事件的患者。

    risk 越大，预测越倾向于更早发生骨转移。
    """
    time_months = np.asarray(time_months)
    event = np.asarray(event)
    risk = np.asarray(risk)
    idx = np.where(event == 1)[0]
    concordant = 0.0
    total = 0
    for i in idx:
        for j in idx:
            if i == j or time_months[i] >= time_months[j]:
                continue
            total += 1
            if risk[i] > risk[j]:
                concordant += 1
            elif risk[i] == risk[j]:
                concordant += 0.5
    return concordant / total if total else float("nan")
