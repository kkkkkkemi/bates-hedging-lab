"""Order-preserving process pool for independent per-path work.

Tasks receive all of their randomness from explicit seeds or pre-drawn arrays,
so serial and parallel runs produce identical artifacts. Worker processes keep
the default of one process, which prevents nested pools.
"""

import os
from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor
from typing import Any

_WORKERS = 1
_THREAD_VARIABLES = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")


def configure(workers: int | None) -> int:
    """Set the pool size. Zero or None selects every logical CPU."""
    global _WORKERS
    _WORKERS = max(1, int(workers) if workers else os.cpu_count() or 1)
    return _WORKERS


def workers() -> int:
    return _WORKERS


def pmap(fn: Callable[[Any], Any], tasks: Iterable[Any]) -> list[Any]:
    """Apply a picklable top-level function to each task and keep task order."""
    jobs = list(tasks)
    count = min(_WORKERS, len(jobs))
    if count <= 1:
        return [fn(job) for job in jobs]
    # Single-threaded BLAS per worker avoids oversubscribing the cores.
    previous = {name: os.environ.get(name) for name in _THREAD_VARIABLES}
    for name in _THREAD_VARIABLES:
        os.environ[name] = "1"
    try:
        with ProcessPoolExecutor(max_workers=count) as pool:
            return list(pool.map(fn, jobs))
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
