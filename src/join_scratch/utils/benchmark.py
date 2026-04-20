"""Benchmark utilities: timing and memory measurement helpers."""

import gc
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone


def _rss_mib() -> float:
    """Return current process RSS memory in MiB.

    Uses ``psutil`` when available (most accurate, cross-platform).
    Falls back to ``/proc/self/status`` on Linux (VmRSS = current RSS).
    Final fallback is ``resource.ru_maxrss`` (peak RSS, not current).
    """
    try:
        import psutil

        return psutil.Process().memory_info().rss / 1024 / 1024
    except ImportError:
        pass

    if sys.platform == "linux":
        try:
            with open("/proc/self/status") as fh:
                for line in fh:
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1]) / 1024  # kB → MiB
        except OSError:
            pass

    # Last resort: ru_maxrss (peak RSS, not current — may underestimate deltas)
    import resource

    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return usage / 1024 / 1024  # macOS: bytes → MiB
    return usage / 1024  # Linux kB → MiB


@dataclass
class BenchmarkResult:
    label: str
    source_shape: tuple[int, int]
    elapsed_s: float
    rss_delta_mib: float
    notes: str = ""


def _time_call(fn, *args, **kwargs) -> tuple[float, float]:
    """Run fn(*args, **kwargs) and return (elapsed_s, rss_delta_mib)."""
    gc.collect()
    rss_before = _rss_mib()
    t0 = time.perf_counter()
    fn(*args, **kwargs)
    return time.perf_counter() - t0, _rss_mib() - rss_before


def render_report(
    results: list[BenchmarkResult],
    title: str = "Benchmark Report",
    timestamp: datetime | None = None,
) -> str:
    """Format a plain-text benchmark table.

    Parameters
    ----------
    results:
        List of benchmark results to include in the report.
    title:
        Title line of the report.
    timestamp:
        Datetime to embed in the report header.  Defaults to the current UTC
        time when the function is called.
    """
    if timestamp is None:
        timestamp = datetime.now(tz=timezone.utc)
    ts_str = timestamp.strftime("%Y-%m-%dT%H:%M:%SZ")

    col_w = (28, 18, 10, 16, 40)
    header = (
        f"{'Method':<{col_w[0]}} {'Source shape':<{col_w[1]}} "
        f"{'Time (s)':>{col_w[2]}} {'RSS delta (MiB)':>{col_w[3]}} "
        f"{'Notes':<{col_w[4]}}"
    )
    sep = "  ".join("-" * w for w in col_w)

    lines = [
        title,
        "=" * len(title),
        f"Timestamp: {ts_str}",
        "",
        header,
        sep,
    ]

    for r in results:
        shape_str = f"{r.source_shape[0]} x {r.source_shape[1]}"
        lines.append(
            f"{r.label:<{col_w[0]}} {shape_str:<{col_w[1]}} "
            f"{r.elapsed_s:>{col_w[2]}.2f} {r.rss_delta_mib:>{col_w[3]}.1f} "
            f"{r.notes:<{col_w[4]}}"
        )

    lines.append("")
    return "\n".join(lines)
