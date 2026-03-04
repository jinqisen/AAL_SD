from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Protocol


@dataclass(frozen=True)
class TraceOptions:
    schema_version: int = 2
    enable_l3_logging: bool = False
    l3_topk: int = 256
    l3_max_selected: Optional[int] = None


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
        enable_l3 = bool(self.legacy_config.get("enable_l3_logging", False))
        topk = int(self.legacy_config.get("l3_topk", 256) or 256)
        max_selected = self.legacy_config.get("l3_max_selected")
        max_selected_i = None if max_selected is None else int(max_selected)
        trace_options = TraceOptions(
            schema_version=2,
            enable_l3_logging=enable_l3,
            l3_topk=topk,
            l3_max_selected=max_selected_i,
        )
        return ExperimentRuntime(
            experiment_name=str(self.name),
            description=str(self.description),
            legacy_config=self.legacy_config,
            trace_options=trace_options,
        )

