import subprocess
from typing import Dict, Optional


class ResourceMonitor:
    def __init__(
        self,
        total_mem_gb: float = 32.0,
        reserve_gb: float = 6.0,
        per_exp_gb: float = 3.5,
        max_concurrent: int = 3,
    ):
        self.total_mem_gb = total_mem_gb
        self.reserve_gb = reserve_gb
        self.per_exp_gb = per_exp_gb
        self.static_max = min(
            max_concurrent, max(1, int((total_mem_gb - reserve_gb) / per_exp_gb))
        )

    def get_available_slots(self, running_count: int) -> int:
        available_mem = self._get_available_mem_gb()
        dynamic_max = max(1, int((available_mem - self.reserve_gb) / self.per_exp_gb))
        return max(0, min(self.static_max, dynamic_max) - running_count)

    def _get_available_mem_gb(self) -> float:
        import platform as _platform

        system = _platform.system()
        try:
            if system == "Darwin":
                return self._get_available_mem_macos()
            elif system == "Linux":
                return self._get_available_mem_linux()
            else:
                return self.total_mem_gb * 0.3
        except Exception:
            return self.total_mem_gb * 0.3

    def _get_available_mem_macos(self) -> float:
        result = subprocess.run(["vm_stat"], capture_output=True, text=True)
        lines = result.stdout.strip().split("\n")
        page_size = 4096
        for line in lines:
            if "page size of" in line:
                parts = line.split()
                for i, p in enumerate(parts):
                    if p == "of" and i + 1 < len(parts):
                        try:
                            page_size = int(parts[i + 1])
                        except ValueError:
                            pass
                break
        free = inactive = 0
        for line in lines:
            if "Pages free" in line:
                free = int(line.split(":")[1].strip().rstrip("."))
            if "Pages inactive" in line:
                inactive = int(line.split(":")[1].strip().rstrip("."))
        return (free + inactive) * page_size / (1024**3)

    def _get_available_mem_linux(self) -> float:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    kb = int(line.split()[1])
                    return kb / (1024 * 1024)
        mem_free = mem_buffers = mem_cached = 0
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if line.startswith("MemFree:"):
                    mem_free = int(line.split()[1])
                elif line.startswith("Buffers:"):
                    mem_buffers = int(line.split()[1])
                elif line.startswith("Cached:"):
                    mem_cached = int(line.split()[1])
        return (mem_free + mem_buffers + mem_cached) / (1024 * 1024)

    def should_pause(self) -> bool:
        available = self._get_available_mem_gb()
        return available < self.reserve_gb * 1.2

    def compute_budget(
        self,
        avg_exp_hours: float = 3.5,
        branch_exp_hours: float = 1.5,
        max_wall_hours: float = 12.0,
    ) -> Dict:
        slots = self.static_max
        phase_a_budget = max_wall_hours * 0.6
        n_branch = min(2, int(phase_a_budget / branch_exp_hours))
        remaining = phase_a_budget - n_branch * branch_exp_hours
        n_full = min(2, int(remaining / avg_exp_hours))
        phase_b_budget = max_wall_hours * 0.4
        n_verify = min(3, int(phase_b_budget / avg_exp_hours))
        return {
            "phase_a_directions": n_branch + n_full,
            "phase_a_branch": n_branch,
            "phase_a_full": n_full,
            "phase_b_verify_seeds": n_verify,
            "max_concurrent": slots,
            "estimated_hours": (n_branch * branch_exp_hours + n_full * avg_exp_hours)
            / slots
            + n_verify * avg_exp_hours / slots,
        }
