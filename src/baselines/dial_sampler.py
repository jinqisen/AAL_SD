from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np


class DIALStyleSampler:
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

    def _cluster_assignments(self, features: np.ndarray) -> Tuple[np.ndarray, int]:
        n_samples = features.shape[0]
        n_clusters = int(getattr(self.config, "DIAL_N_CLUSTERS", 0) or 0)
        n_clusters = n_clusters if n_clusters > 1 else min(8, max(2, int(np.sqrt(n_samples))))
        if n_clusters > n_samples:
            n_clusters = n_samples
        if n_clusters <= 1:
            return np.zeros((n_samples,), dtype=np.int64), 1
        from sklearn.cluster import KMeans

        km = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
        try:
            import joblib
        except Exception:
            joblib = None
        if joblib is not None:
            with joblib.parallel_backend("threading"):
                labels = km.fit_predict(features)
        else:
            labels = km.fit_predict(features)
        return labels.astype(np.int64), int(n_clusters)

    def rank_samples(self, unlabeled_info: Dict[str, Dict[str, Any]], **kwargs: Any) -> List[Dict[str, Any]]:
        sample_ids = list(unlabeled_info.keys())
        if not sample_ids:
            return []

        uncertainties = {}
        features = []
        feature_ids = []
        for sid in sample_ids:
            info = unlabeled_info.get(sid) or {}
            uncertainties[sid] = self._get_uncertainty(info)
            feat = info.get("feature")
            if feat is not None:
                arr = np.asarray(feat, dtype=np.float32)
                if arr.ndim == 1:
                    features.append(arr)
                    feature_ids.append(sid)

        ranked_ids: List[str] = []
        if len(features) >= 2:
            feats = np.stack(features, axis=0)
            labels, n_clusters = self._cluster_assignments(feats)
            buckets: Dict[int, List[str]] = {i: [] for i in range(n_clusters)}
            for sid, label in zip(feature_ids, labels):
                buckets[int(label)].append(sid)
            for label in buckets:
                buckets[label].sort(key=lambda s: uncertainties.get(s, 0.0), reverse=True)
            remaining = True
            while remaining:
                remaining = False
                for label in range(n_clusters):
                    bucket = buckets[label]
                    if bucket:
                        ranked_ids.append(bucket.pop(0))
                        remaining = True

        covered = set(ranked_ids)
        remaining_ids = [sid for sid in sample_ids if sid not in covered]
        remaining_ids.sort(key=lambda s: uncertainties.get(s, 0.0), reverse=True)
        ranked_ids.extend(remaining_ids)

        ranked: List[Dict[str, Any]] = []
        for sid in ranked_ids:
            ranked.append(
                {
                    "sample_id": sid,
                    "final_score": float(uncertainties.get(sid, 0.0)),
                    "uncertainty": float(uncertainties.get(sid, 0.0)),
                }
            )
        return ranked
