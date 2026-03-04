
import numpy as np

class EntropySampler:
    """基于熵的不确定性采样基线"""
    
    def __init__(self, config=None):
        pass

    def _calculate_entropy(self, prob_map: np.ndarray) -> float:
        """计算像素级熵的平均值"""
        eps = 1e-10
        # prob_map: (C, H, W) or (N, C, H, W)
        # If input is single sample features/probs
        if prob_map.ndim == 3:
             entropy = -np.sum(prob_map * np.log2(prob_map + eps), axis=0)
             return np.mean(entropy)
        return 0.0

    def rank_samples(self, unlabeled_info: dict, **kwargs) -> list:
        """根据不确定性（熵）对样本进行排序。"""
        scores = []
        for sample_id, info in unlabeled_info.items():
            if info is None:
                scores.append({"sample_id": sample_id, "final_score": 0.0})
                continue

            if info.get("uncertainty_score") is not None:
                scores.append(
                    {
                        "sample_id": sample_id,
                        "final_score": float(info.get("uncertainty_score") or 0.0),
                    }
                )
                continue

            probs = info.get("prob_map")
            if probs is None:
                probs = info.get("probs")

            if probs is not None:
                uncertainty = self._calculate_entropy(probs)
                scores.append({"sample_id": sample_id, "final_score": float(uncertainty)})
            else:
                scores.append({"sample_id": sample_id, "final_score": 0.0})
        
        scores.sort(key=lambda x: x["final_score"], reverse=True)
        return scores
