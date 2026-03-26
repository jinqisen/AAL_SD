from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Protocol


@dataclass(frozen=True)
class TraceOptions:
    schema_version: int = 3
    enable_l3_selection_logging: bool = False
    enable_agent_prompt_logging: bool = False
    l3_topk: int = 256
    l3_max_selected: Optional[int] = None
    enable_score_snapshot_logging: bool = False
    score_snapshot_boundary_window: int = 64
    score_snapshot_max_pool_items: Optional[int] = None


@dataclass(frozen=True)
class ExperimentRuntime:
    experiment_name: str
    description: str
    legacy_config: Mapping[str, Any]
    trace_options: TraceOptions


class ExperimentSpec(Protocol):
    name: str
    description: str

    def build(self, config: Any) -> ExperimentRuntime: ...


@dataclass(frozen=True)
class LegacyExperimentSpec:
    name: str
    description: str
    legacy_config: Mapping[str, Any]

    def build(self, config: Any) -> ExperimentRuntime:
        use_agent = bool(self.legacy_config.get("use_agent", False))
        enable_l3_selection = bool(
            self.legacy_config.get("enable_l3_selection_logging", use_agent)
        )
        enable_agent_prompt = bool(
            self.legacy_config.get("enable_agent_prompt_logging", use_agent)
        )
        enable_score_snapshot = bool(
            self.legacy_config.get(
                "enable_score_snapshot_logging", enable_l3_selection
            )
        )
        topk = int(self.legacy_config.get("l3_topk", 256) or 256)
        max_selected = self.legacy_config.get("l3_max_selected")
        max_selected_i = None if max_selected is None else int(max_selected)
        boundary_window = int(
            self.legacy_config.get("score_snapshot_boundary_window", 64) or 64
        )
        max_pool_items = self.legacy_config.get("score_snapshot_max_pool_items")
        max_pool_items_i = (
            None if max_pool_items is None else int(max_pool_items)
        )
        trace_options = TraceOptions(
            schema_version=3,
            enable_l3_selection_logging=enable_l3_selection,
            enable_agent_prompt_logging=enable_agent_prompt,
            l3_topk=topk,
            l3_max_selected=max_selected_i,
            enable_score_snapshot_logging=enable_score_snapshot,
            score_snapshot_boundary_window=boundary_window,
            score_snapshot_max_pool_items=max_pool_items_i,
        )
        return ExperimentRuntime(
            experiment_name=str(self.name),
            description=str(self.description),
            legacy_config=self.legacy_config,
            trace_options=trace_options,
        )
