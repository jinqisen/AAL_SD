from typing import Dict, List, Tuple


class ConvergenceDetector:
    def __init__(
        self, target_miou: float = 0.74, patience: int = 3, max_iterations: int = 10
    ):
        self.target_miou = target_miou
        self.patience = patience
        self.max_iterations = max_iterations
        self.history: List[Tuple[int, float]] = []

    def update(self, iteration: int, best_miou: float) -> Dict:
        self.history.append((iteration, best_miou))
        if best_miou >= self.target_miou:
            return {
                "action": "STOP",
                "reason": "target_reached",
                "best_miou": best_miou,
            }
        if len(self.history) >= self.patience:
            recent = [m for _, m in self.history[-self.patience :]]
            if len(self.history) > self.patience:
                baseline = self.history[-(self.patience + 1)][1]
            else:
                baseline = recent[0]
            if all(m <= baseline + 0.002 for m in recent):
                return {
                    "action": "STOP",
                    "reason": "convergence_plateau",
                    "best_miou": max(m for _, m in self.history),
                }
        if iteration >= self.max_iterations:
            return {
                "action": "STOP",
                "reason": "max_iterations",
                "best_miou": max(m for _, m in self.history),
            }
        if len(self.history) >= 4:
            deltas = [self.history[i][1] - self.history[i - 1][1] for i in range(-3, 0)]
            if max(deltas) < 0.001:
                return {
                    "action": "WARN",
                    "reason": "diminishing_returns",
                    "avg_delta": sum(deltas) / len(deltas),
                }
        return {"action": "CONTINUE"}
