from __future__ import annotations

from typing import Any

from experiments.ablation_config import ABLATION_SETTINGS, EXPERIMENT_NAME_ALIASES, build_spec_from_legacy_dict
from experiments.specs.types import ExperimentRuntime, ExperimentSpec


def get_experiment_spec(experiment_name: str) -> ExperimentSpec:
    requested = str(experiment_name)
    canonical = str(EXPERIMENT_NAME_ALIASES.get(requested, requested))
    cfg = ABLATION_SETTINGS.get(canonical)
    if not isinstance(cfg, dict):
        raise ValueError(f"Unknown experiment: {experiment_name}")
    return build_spec_from_legacy_dict(requested, cfg)


def build_experiment_runtime(experiment_name: str, config: Any) -> ExperimentRuntime:
    spec = get_experiment_spec(experiment_name)
    return spec.build(config)
