from __future__ import annotations

from typing import Any, Dict, List

import numpy as np


class WangStyleSampler:
    def __init__(self, config: Any | None = None) -> None:
        self.config = config

    def _get_uncertainty(self, info: Dict[str, Any]) -> float:
        if info is None:
            return 0.0
        if "uncertainty_score" in info and info.get("uncertainty_score") is not None:
            return float(info.get("uncertainty_score") or 0.0)
        prob_map = info.get("prob_map")
        if prob_map is None:
            return 0.0
        try:
            prob = np.asarray(prob_map, dtype=np.float64)
            eps = 1e-12
            entropy = -np.sum(prob * np.log(prob + eps), axis=0)
            return float(np.mean(entropy))
        except Exception:
            return 0.0

    def rank_samples(
        self,
        unlabeled_info: Dict[str, Dict[str, Any]],
        labeled_features: np.ndarray | None = None,
        current_round: int | None = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        sample_ids = list(unlabeled_info.keys())
        if not sample_ids:
            return []

        warmup_rounds = int(getattr(self.config, "WANG_WARMUP_ROUNDS", 0) or 2)
        candidate_ratio = float(getattr(self.config, "WANG_CANDIDATE_RATIO", 0.0) or 0.3)
        min_candidates = int(getattr(self.config, "WANG_MIN_CANDIDATES", 0) or 10)
        round_idx = int(current_round or 1)

        uncertainties = {sid: self._get_uncertainty(unlabeled_info.get(sid) or {}) for sid in sample_ids}
        sorted_by_uncertainty = sorted(sample_ids, key=lambda s: uncertainties.get(s, 0.0), reverse=True)

        if labeled_features is None or labeled_features.size == 0 or round_idx <= warmup_rounds:
            return [
                {
                    "sample_id": sid,
                    "final_score": float(uncertainties.get(sid, 0.0)),
                    "uncertainty": float(uncertainties.get(sid, 0.0)),
                }
                for sid in sorted_by_uncertainty
            ]

        candidate_count = max(min_candidates, int(len(sample_ids) * candidate_ratio))
        candidate_count = min(candidate_count, len(sample_ids))
        candidate_ids = sorted_by_uncertainty[:candidate_count]

        features = []
        valid_ids = []
        for sid in candidate_ids:
            feat = (unlabeled_info.get(sid) or {}).get("feature")
            if feat is None:
                continue
            arr = np.asarray(feat, dtype=np.float32)
            if arr.ndim != 1:
                continue
            features.append(arr)
            valid_ids.append(sid)

        if not valid_ids:
            return [
                {
                    "sample_id": sid,
                    "final_score": float(uncertainties.get(sid, 0.0)),
                    "uncertainty": float(uncertainties.get(sid, 0.0)),
                }
                for sid in sorted_by_uncertainty
            ]

        cand_features = np.stack(features, axis=0)
        labeled = np.asarray(labeled_features, dtype=np.float32)
        diffs = cand_features[:, None, :] - labeled[None, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        min_d = np.min(dists, axis=1)
        rep_scores = {sid: float(score) for sid, score in zip(valid_ids, min_d)}

        ranked_candidates = sorted(valid_ids, key=lambda s: rep_scores.get(s, 0.0), reverse=True)
        remaining_ids = [sid for sid in sorted_by_uncertainty if sid not in set(ranked_candidates)]
        ranked_ids = ranked_candidates + remaining_ids

        ranked: List[Dict[str, Any]] = []
        for sid in ranked_ids:
            ranked.append(
                {
                    "sample_id": sid,
                    "final_score": float(rep_scores.get(sid, uncertainties.get(sid, 0.0))),
                    "uncertainty": float(uncertainties.get(sid, 0.0)),
                    "knowledge_gain": float(rep_scores.get(sid, 0.0)) if sid in rep_scores else None,
                }
            )
        return ranked
