from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Sequence, Final

import numpy as np


@dataclass(frozen=True)
class SamplerBuildResult:
    sampler: Any
    sampler_type: str
    rollback_config: Optional[dict]
    score_normalization: bool


def build_sampler(config: Any, exp_config: Mapping[str, Any]) -> SamplerBuildResult:
    from baselines.bald_sampler import BALDSampler
    from baselines.coreset_sampler import CoresetSampler
    from baselines.dial_sampler import DIALStyleSampler
    from baselines.entropy_sampler import EntropySampler
    from baselines.llm_rs_sampler import LLMRandomSampler
    from baselines.llm_us_sampler import LLMUncertaintySampler
    from baselines.oracle_hardpos_sampler import OracleHardPosSampler
    from baselines.random_sampler import RandomSampler
    from baselines.wang_sampler import WangStyleSampler
    from core.sampler import ADKUCSSampler

    sampler_type = str(exp_config["sampler_type"])
    score_normalization = bool(exp_config.get("score_normalization", True))

    def _build_random() -> tuple[Any, Optional[dict]]:
        return RandomSampler(config), None

    def _build_entropy() -> tuple[Any, Optional[dict]]:
        return EntropySampler(config), None

    def _build_coreset() -> tuple[Any, Optional[dict]]:
        return CoresetSampler(config), None

    def _build_bald() -> tuple[Any, Optional[dict]]:
        return BALDSampler(config), None

    def _build_llm_us() -> tuple[Any, Optional[dict]]:
        return LLMUncertaintySampler(config), None

    def _build_llm_rs() -> tuple[Any, Optional[dict]]:
        return LLMRandomSampler(config), None

    def _build_dial() -> tuple[Any, Optional[dict]]:
        return DIALStyleSampler(config), None

    def _build_wang() -> tuple[Any, Optional[dict]]:
        return WangStyleSampler(config), None

    def _build_oracle_hardpos() -> tuple[Any, Optional[dict]]:
        oracle_cfg = exp_config.get("oracle_hardpos", {})
        return OracleHardPosSampler(
            config,
            replace_ratio=float(oracle_cfg.get("replace_ratio", 0.25)),
            hardpos_percentile=float(oracle_cfg.get("hardpos_percentile", 0.0)),
        ), None

    def _build_ad_kucs() -> tuple[Any, Optional[dict]]:
        rollback_config = exp_config.get("rollback_config") or {
            "mode": "adaptive_threshold",
            "std_factor": 1.5,
            "tau_min": 0.005,
        }
        sampler = ADKUCSSampler(
            device=config.DEVICE,
            alpha=config.ALPHA,
            score_normalization=score_normalization,
            feature_num_workers=getattr(config, "FEATURE_NUM_WORKERS", 0),
            feature_persistent_workers=getattr(config, "FEATURE_PERSISTENT_WORKERS", False),
            feature_prefetch_factor=getattr(config, "FEATURE_PREFETCH_FACTOR", 2),
            feature_pin_memory=getattr(config, "FEATURE_PIN_MEMORY", False),
        )
        return sampler, rollback_config

    sampler_builders: Final[dict[str, Callable[[], tuple[Any, Optional[dict]]]]] = {
        "random": _build_random,
        "entropy": _build_entropy,
        "coreset": _build_coreset,
        "bald": _build_bald,
        "llm_us": _build_llm_us,
        "llm_rs": _build_llm_rs,
        "dial": _build_dial,
        "wang": _build_wang,
        "oracle_hardpos": _build_oracle_hardpos,
        "ad_kucs": _build_ad_kucs,
    }
    builder = sampler_builders.get(sampler_type)
    if builder is None:
        raise ValueError(f"Unknown sampler_type: {sampler_type}")
    sampler, rollback_config = builder()
    if hasattr(sampler, "configure_from_exp"):
        sampler.configure_from_exp(exp_config)

    return SamplerBuildResult(
        sampler=sampler,
        sampler_type=sampler_type,
        rollback_config=rollback_config,
        score_normalization=score_normalization,
    )


@dataclass(frozen=True)
class LegacySelectionPostprocessor:
    post_cfg: Mapping[str, Any]
    constraints: Mapping[str, Any] | None
    select_diverse: Callable[
        [Sequence[Mapping[str, Any]], Mapping[str, Any], int, Mapping[str, Any]],
        list,
    ]
    get_round: Callable[[], int | None]

    def apply(
        self,
        ranked: Sequence[Mapping[str, Any]],
        unlabeled_info: Mapping[str, Any],
        n_samples: int,
    ) -> tuple[list[Any], dict]:
        if not ranked:
            return [], {"applied": False, "reason": "empty_ranked"}

        mode = str(self.post_cfg.get("mode", "none")).strip().lower() or "none"
        postprocess_enabled_by_mode: Final[dict[str, bool]] = {"none": False, "fps_feature": True}
        enabled = postprocess_enabled_by_mode.get(mode, False)
        if not enabled:
            return [item.get("sample_id") for item in ranked], {"applied": False}

        candidate_multiplier = int(self.post_cfg.get("candidate_multiplier", 5))
        top_m = max(int(n_samples), int(candidate_multiplier) * int(n_samples))
        ranked_trimmed = list(ranked[:top_m])

        constraints = self.constraints if isinstance(self.constraints, dict) else None
        constraints_mode = {True: "pred_pos_area_quota", False: "none"}[
            bool(constraints is not None and bool(constraints.get("use_pred_pos_area", False)))
        ]
        constraint_selected: dict = {"pos_selected": 0, "neg_selected": 0}

        def _select_none() -> list:
            return self.select_diverse(ranked_trimmed, unlabeled_info, int(n_samples), self.post_cfg)

        def _select_pred_pos_area_quota() -> list:
            pos_class = int(constraints.get("pos_class", 1))  # type: ignore[union-attr]
            pos_threshold = float(constraints.get("pos_threshold", 0.5))  # type: ignore[union-attr]
            pos_area_min = float(constraints.get("pos_area_min", 0.001))  # type: ignore[union-attr]
            pos_ratio = float(constraints.get("pos_quota_ratio", 0.5))  # type: ignore[union-attr]

            pos_items = []
            neg_items = []
            for item in ranked_trimmed:
                sid = item.get("sample_id")
                info = unlabeled_info.get(sid, {}) if isinstance(unlabeled_info, dict) else {}
                pos_area = 0.0
                if isinstance(info, dict) and info.get("pos_area") is not None:
                    try:
                        pos_area = float(info.get("pos_area") or 0.0)
                    except Exception:
                        pos_area = 0.0
                else:
                    prob_map = info.get("prob_map") if isinstance(info, dict) else None
                    if prob_map is not None:
                        try:
                            channel = prob_map[pos_class]
                            pos_area = float(np.mean(channel > pos_threshold))
                        except Exception:
                            pos_area = 0.0
                if pos_area >= pos_area_min:
                    pos_items.append(item)
                else:
                    neg_items.append(item)

            quota = int(round(float(n_samples) * float(pos_ratio)))
            quota = max(0, min(int(n_samples), quota))
            selected_items_local = self.select_diverse(pos_items, unlabeled_info, quota, self.post_cfg)
            constraint_selected["pos_selected"] = int(len(selected_items_local))

            remaining = int(n_samples) - len(selected_items_local)
            if remaining > 0:
                rest_pool = neg_items + [i for i in pos_items if i not in selected_items_local]
                rest_selected = self.select_diverse(rest_pool, unlabeled_info, remaining, self.post_cfg)
                constraint_selected["neg_selected"] = int(len(rest_selected))
                selected_items_local.extend(rest_selected)
            return selected_items_local

        constraint_selectors: Final[dict[str, Callable[[], list]]] = {
            "none": _select_none,
            "pred_pos_area_quota": _select_pred_pos_area_quota,
        }
        selected_items = constraint_selectors[constraints_mode]()
        pos_selected = int(constraint_selected["pos_selected"])
        neg_selected = int(constraint_selected["neg_selected"])

        selected_ids = [item["sample_id"] for item in selected_items if "sample_id" in item]

        if len(selected_ids) < int(n_samples):
            seen = set(selected_ids)
            for item in ranked_trimmed:
                sid = item.get("sample_id")
                if sid not in seen:
                    selected_ids.append(sid)
                    seen.add(sid)
                if len(selected_ids) >= int(n_samples):
                    break

        remaining_ids = [
            item.get("sample_id")
            for item in ranked_trimmed
            if item.get("sample_id") not in set(selected_ids)
        ]

        meta: dict = {
            "applied": True,
            "mode": mode,
            "candidate_multiplier": int(candidate_multiplier),
            "candidate_size": int(len(ranked_trimmed)),
        }
        def _meta_none():
            return None

        def _meta_pred_pos_area_quota():
            meta.update(
                {
                    "pos_quota_ratio": float(constraints.get("pos_quota_ratio", 0.5)),  # type: ignore[union-attr]
                    "pos_selected": int(pos_selected),
                    "neg_selected": int(neg_selected),
                    "pos_area_min": float(constraints.get("pos_area_min", 0.001)),  # type: ignore[union-attr]
                    "pos_threshold": float(constraints.get("pos_threshold", 0.5)),  # type: ignore[union-attr]
                }
            )
            return None

        meta_updaters: Final[dict[str, Callable[[], None]]] = {
            "none": _meta_none,
            "pred_pos_area_quota": _meta_pred_pos_area_quota,
        }
        meta_updaters.get(constraints_mode, _meta_none)()
        return list(selected_ids) + list(remaining_ids), meta


def build_selection_postprocessor(pipeline: Any) -> LegacySelectionPostprocessor | None:
    exp_config = getattr(pipeline, "exp_config", None)
    if not isinstance(exp_config, dict):
        return None
    protocol = exp_config.get("acquisition_protocol")
    protocol = protocol if isinstance(protocol, dict) else {}
    mode = str(protocol.get("diversity_postprocess", "none") or "none").strip().lower()
    if mode != "fps_feature":
        return None
    candidate_multiplier = int(protocol.get("candidate_multiplier", 5) or 5)
    constraints = exp_config.get("candidate_constraints")
    post_cfg = {"mode": "fps_feature", "candidate_multiplier": candidate_multiplier}
    return LegacySelectionPostprocessor(
        post_cfg=post_cfg,
        constraints=constraints if isinstance(constraints, dict) else None,
        select_diverse=pipeline._select_diverse_items,
        get_round=lambda: getattr(pipeline, "current_round", None),
    )
