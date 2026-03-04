
import numpy as np
from sklearn.metrics import auc

def calculate_alc(
    performance_history: list,
    budget_history: list,
    *,
    total_budget: int | None = None,
    pad_to_total_budget: bool = True,
) -> float:
    if not performance_history or not budget_history:
        return 0.0
    if len(performance_history) != len(budget_history):
        n = min(len(performance_history), len(budget_history))
        performance_history = performance_history[:n]
        budget_history = budget_history[:n]
    if len(budget_history) < 2:
        return 0.0

    budgets = np.asarray(budget_history, dtype=float)
    perfs = np.asarray(performance_history, dtype=float)

    order = np.argsort(budgets)
    budgets = budgets[order]
    perfs = perfs[order]

    unique_budgets = []
    unique_perfs = []
    last_b = None
    for b, p in zip(budgets.tolist(), perfs.tolist()):
        if last_b is None or b != last_b:
            unique_budgets.append(b)
            unique_perfs.append(p)
            last_b = b
        else:
            if unique_perfs:
                unique_perfs[-1] = max(float(unique_perfs[-1]), float(p))

    budgets = np.asarray(unique_budgets, dtype=float)
    perfs = np.asarray(unique_perfs, dtype=float)

    if total_budget is not None and int(total_budget) > 0:
        denom = float(int(total_budget))
        x = np.clip(budgets / denom, 0.0, 1.0)
        if pad_to_total_budget and x.size > 0 and float(x[-1]) < 1.0:
            x = np.append(x, 1.0)
            perfs = np.append(perfs, float(perfs[-1]))
        if x.size < 2:
            return 0.0
        return float(auc(x, perfs))

    span = float(np.max(budgets) - np.min(budgets))
    if span <= 0:
        return 0.0
    x = (budgets - float(np.min(budgets))) / span
    return float(auc(x, perfs))

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 2) -> dict:
    y_true_flat = np.asarray(y_true).reshape(-1)
    y_pred_flat = np.asarray(y_pred).reshape(-1)

    k = int(num_classes)
    if k <= 0:
        return {"mIoU": 0.0, "f1_score": 0.0}

    y_true_flat = np.clip(y_true_flat.astype(np.int64, copy=False), 0, k - 1)
    y_pred_flat = np.clip(y_pred_flat.astype(np.int64, copy=False), 0, k - 1)
    indices = y_true_flat * k + y_pred_flat
    conf = np.bincount(indices, minlength=k * k).reshape(k, k).astype(np.float64)

    tp = np.diag(conf)
    fp = conf.sum(axis=0) - tp
    fn = conf.sum(axis=1) - tp

    iou_denom = tp + fp + fn
    f1_denom = 2.0 * tp + fp + fn

    iou = np.where(iou_denom > 0.0, tp / iou_denom, 1.0)
    f1 = np.where(f1_denom > 0.0, (2.0 * tp) / f1_denom, 1.0)

    return {"mIoU": float(np.mean(iou)), "f1_score": float(np.mean(f1))}
