"""Run one reproducible experiment configuration and render its validation report."""

import argparse
import importlib.metadata
import importlib.util
from pathlib import Path
from time import perf_counter

import yaml

from quantproj import experiments as ex
from quantproj import parallel
from quantproj.models import Bates
from quantproj.reporting import render


def _run(config: dict, output: Path) -> None:
    if config["greeks"]["aad"] and importlib.util.find_spec("jax") is None:
        raise SystemExit("The selected configuration requires JAX. Install the project with the aad extra.")
    output.mkdir(parents=True, exist_ok=True)
    print(f"Using {parallel.configure(config.get('workers', 0))} worker processes", flush=True)
    seed = config["seed"]
    model = Bates(**config.get("market", {}))
    times = config["surface"]["tenors"]
    log_k = config["surface"]["log_moneyness"]
    modes = config["surface"]["cos_modes"]
    noise = config["surface"]["noise"]
    timings = {}
    tasks = [
        ("pricing", lambda: ex.pricing_experiment(output, model, config["pricing"], seed + 1)),
        (
            "calibration",
            lambda: ex.calibration_experiment(
                output, model, config["calibration"], seed + 2, times, log_k, modes
            ),
        ),
        (
            "hedging",
            lambda: ex.hedge_experiment(
                output, model, config["hedging"], seed + 3, times, log_k, noise, modes
            ),
        ),
        ("greeks", lambda: ex.greek_experiment(output, model, config["greeks"], seed + 4)),
        (
            "pnl",
            lambda: ex.pnl_experiment(output, model, config["pnl"], seed + 5, times, log_k, noise, modes),
        ),
        ("commodity", lambda: ex.commodity_experiment(output, config["commodity"], seed + 6)),
    ]
    for name, callback in tasks:
        print(f"Running {name}", flush=True)
        start = perf_counter()
        callback()
        timings[name] = perf_counter() - start
        print(f"Finished {name} in {timings[name]:.1f}s", flush=True)
    dependencies = {}
    for name in ("numpy", "scipy", "matplotlib", "PyYAML", "reportlab", "jax"):
        try:
            dependencies[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            dependencies[name] = None
    render(output, config)
    ex.write_json(
        output, "manifest", {"configuration": config, "seconds": timings, "dependencies": dependencies}
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("experiments/full.yaml"))
    parser.add_argument("--output", type=Path, default=Path("output/full"))
    parser.add_argument(
        "--render-only", action="store_true", help="Render again from existing CSV artifacts."
    )
    parser.add_argument(
        "--workers", type=int, help="Worker processes; 0 uses every logical CPU, 1 runs serially."
    )
    parsed = parser.parse_args()
    config = yaml.safe_load(parsed.config.read_text(encoding="utf-8"))
    if parsed.workers is not None:
        config["workers"] = parsed.workers
    if parsed.render_only:
        render(parsed.output, config)
    else:
        _run(config, parsed.output)
    print(f"Report: {parsed.output / 'model_validation.pdf'}")


if __name__ == "__main__":
    main()
