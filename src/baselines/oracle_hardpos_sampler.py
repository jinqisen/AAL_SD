"""Oracle Hard-Positive Sampler for Phase 1 causal chain experiments.

Takes the entropy-ranked batch and replaces the weakest-foreground tail with
oracle hard-positive samples (highest GT foreground fraction from the
unlabeled pool).  The replacement ratio is a configurable parameter
(e.g. 0.10, 0.25, 0.50) so we can measure whether the blind-spot gap
is exploitable at all.

This sampler is *not* a practical AL strategy — it uses ground-truth labels
and exists solely to close the causal link between U's blind spot and
downstream mIoU.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import h5py
import numpy as np

from baselines.entropy_sampler import EntropySampler

logger = logging.getLogger(__name__)


class OracleHardPosSampler:
    """Entropy + oracle hard-positive injection.

    Parameters
    ----------
    config : Any
        Global experiment config (unused beyond forwarding to EntropySampler).
    replace_ratio : float
        Fraction of the query batch to replace with oracle hard-positives.
        Must be in (0, 1].
    hardpos_percentile : float
        Percentile threshold (0–100) on gt_positive_frac to define
        "hard-positive".  Samples with gt_frac >= this percentile of the
        *current unlabeled pool* are eligible for oracle injection.
        Default 0 means all foreground-containing samples are candidates,
        ranked by gt_frac descending.
    """

    def __init__(
        self,
        config: Any = None,
        replace_ratio: float = 0.25,
        hardpos_percentile: float = 0.0,
    ):
        self._entropy_sampler = EntropySampler(config)
        self.replace_ratio = float(replace_ratio)
        self.hardpos_percentile = float(hardpos_percentile)
        # gt_frac_map: {sample_id (dataset index) -> gt_positive_frac}
        # Set externally via set_gt_frac_map() before first rank_samples call.
        self._gt_frac_map: Dict[Any, float] = {}

    # ------------------------------------------------------------------
    # Pipeline integration helpers
    # ------------------------------------------------------------------

    def set_gt_frac_map(self, gt_frac_map: Dict[Any, float]) -> None:
        """Inject precomputed GT foreground fractions keyed by dataset index."""
        self._gt_frac_map = dict(gt_frac_map)
        logger.info(
            "OracleHardPosSampler: gt_frac_map set with %d entries", len(self._gt_frac_map)
        )

    def configure_from_exp(self, exp_config: dict) -> None:
        """Read oracle-specific knobs from the ablation config."""
        oracle_cfg = exp_config.get("oracle_hardpos", {})
        if oracle_cfg:
            if "replace_ratio" in oracle_cfg:
                self.replace_ratio = float(oracle_cfg["replace_ratio"])
            if "hardpos_percentile" in oracle_cfg:
                self.hardpos_percentile = float(oracle_cfg["hardpos_percentile"])

    # ------------------------------------------------------------------
    # Core ranking
    # ------------------------------------------------------------------

    def rank_samples(self, unlabeled_info: dict, **kwargs) -> list:
        """Return a ranked list with oracle hard-positive injection.

        1. Run entropy ranking on the full unlabeled pool.
        2. Identify oracle hard-positive candidates (highest gt_frac).
        3. In the top-QUERY_SIZE window, replace the tail (lowest gt_frac)
           with oracle hard-positives that were *not* already selected.
        4. Return the full re-ordered ranking so the pipeline's
           ``ranked[:n_samples]`` slicing picks up the injected samples.
        """
        # Step 1: entropy ranking
        entropy_ranked: List[dict] = self._entropy_sampler.rank_samples(
            unlabeled_info, **kwargs
        )

        if not self._gt_frac_map:
            logger.warning(
                "OracleHardPosSampler: gt_frac_map is empty — falling back to pure entropy"
            )
            return entropy_ranked

        if not entropy_ranked:
            return entropy_ranked

        # Determine effective query size from kwargs or use full ranking length.
        # The pipeline slices ranked[:n_samples], so we operate on that window.
        query_size = int(kwargs.get("query_size", 0)) or len(entropy_ranked)

        n_replace = max(1, int(round(query_size * self.replace_ratio)))

        # Step 2: build oracle hard-positive candidate pool
        # Only consider samples currently in the unlabeled pool (keys of unlabeled_info).
        pool_ids = set(unlabeled_info.keys())
        gt_fracs_in_pool = {
            sid: frac
            for sid, frac in self._gt_frac_map.items()
            if sid in pool_ids and frac > 0.0
        }

        if not gt_fracs_in_pool:
            logger.info(
                "OracleHardPosSampler: no positive-frac samples in unlabeled pool — pure entropy"
            )
            return entropy_ranked

        # Apply percentile threshold if set
        if self.hardpos_percentile > 0:
            fracs = np.array(list(gt_fracs_in_pool.values()))
            threshold = float(np.percentile(fracs, self.hardpos_percentile))
            gt_fracs_in_pool = {
                sid: frac for sid, frac in gt_fracs_in_pool.items() if frac >= threshold
            }

        # Sort oracle candidates by gt_frac descending
        oracle_candidates = sorted(
            gt_fracs_in_pool.items(), key=lambda x: x[1], reverse=True
        )

        # Step 3: identify which entropy-selected samples to replace
        top_window = entropy_ranked[:query_size]
        tail_section = entropy_ranked[query_size:]

        # Already-selected IDs in top window
        top_ids = {item["sample_id"] for item in top_window}

        # Oracle candidates not already in the top window
        injectable = [
            (sid, frac) for sid, frac in oracle_candidates if sid not in top_ids
        ]

        actual_inject = min(n_replace, len(injectable))
        if actual_inject == 0:
            logger.info(
                "OracleHardPosSampler: all oracle candidates already in entropy top-K"
            )
            return entropy_ranked

        # Samples to inject (highest gt_frac first)
        inject_ids = {injectable[i][0] for i in range(actual_inject)}
        inject_items = [
            {
                "sample_id": sid,
                "final_score": float(unlabeled_info[sid].get("uncertainty_score", 0.0))
                if isinstance(unlabeled_info.get(sid), dict)
                else 0.0,
                "_oracle_injected": True,
                "_gt_frac": frac,
            }
            for sid, frac in injectable[:actual_inject]
        ]

        # Remove the weakest-foreground tail from the top window to make room.
        # "Weakest foreground" = lowest gt_frac among top_window samples.
        top_with_frac = []
        for item in top_window:
            sid = item["sample_id"]
            frac = self._gt_frac_map.get(sid, 0.0)
            top_with_frac.append((item, frac))

        # Sort ascending by gt_frac — the first `actual_inject` are evicted
        top_with_frac.sort(key=lambda x: x[1])
        evicted = [pair[0] for pair in top_with_frac[:actual_inject]]
        kept = [pair[0] for pair in top_with_frac[actual_inject:]]

        # Reconstruct: kept (sorted by original entropy score desc) + injected
        kept.sort(key=lambda x: float(x.get("final_score", 0.0)), reverse=True)

        new_top = kept + inject_items
        # Put evicted back into the tail
        evicted_set = {item["sample_id"] for item in evicted}
        # Remove any inject_ids from tail (they were in the tail before injection)
        tail_cleaned = [
            item
            for item in tail_section
            if item["sample_id"] not in inject_ids
        ]

        result = new_top + evicted + tail_cleaned

        # Log summary
        inject_fracs = [frac for _, frac in injectable[:actual_inject]]
        evicted_fracs = [self._gt_frac_map.get(item["sample_id"], 0.0) for item in evicted]
        logger.info(
            "OracleHardPosSampler: injected %d/%d oracle hard-positives "
            "(replace_ratio=%.2f, gt_frac range=[%.4f, %.4f]), "
            "evicted %d samples (gt_frac range=[%.4f, %.4f])",
            actual_inject,
            query_size,
            self.replace_ratio,
            min(inject_fracs) if inject_fracs else 0.0,
            max(inject_fracs) if inject_fracs else 0.0,
            len(evicted),
            min(evicted_fracs) if evicted_fracs else 0.0,
            max(evicted_fracs) if evicted_fracs else 0.0,
        )

        return result


def compute_gt_frac_map_from_pool(
    pool_df,
    dataset_index_map: Dict[str, int],
) -> Dict[int, float]:
    """Compute GT foreground fraction for every sample in a pool DataFrame.

    Parameters
    ----------
    pool_df : pd.DataFrame
        Must contain columns ``sample_id`` and ``mask_path``.
    dataset_index_map : dict
        Mapping from sample_id (string) to dataset index (int).

    Returns
    -------
    dict
        ``{dataset_index: gt_positive_frac}`` where gt_positive_frac ∈ [0, 1].
    """
    gt_frac: Dict[int, float] = {}
    for _, row in pool_df.iterrows():
        sid = str(row["sample_id"])
        idx = dataset_index_map.get(sid)
        if idx is None:
            continue
        mask_path = str(row["mask_path"])
        if not os.path.isfile(mask_path):
            logger.warning("compute_gt_frac_map: mask not found: %s", mask_path)
            gt_frac[idx] = 0.0
            continue
        try:
            with h5py.File(mask_path, "r") as f:
                mask = f["mask"][()]
            frac = float(np.mean(np.asarray(mask) > 0))
        except Exception as exc:
            logger.warning("compute_gt_frac_map: failed to read %s: %s", mask_path, exc)
            frac = 0.0
        gt_frac[idx] = frac
    logger.info(
        "compute_gt_frac_map: computed fracs for %d samples (%.1f%% positive)",
        len(gt_frac),
        100.0 * sum(1 for v in gt_frac.values() if v > 0) / max(len(gt_frac), 1),
    )
    return gt_frac
