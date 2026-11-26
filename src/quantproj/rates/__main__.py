"""Run and report the rates model-validation experiments."""

import argparse
from pathlib import Path

import yaml

from quantproj.rates.experiments import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("experiments/rates/quick.yaml"))
    parser.add_argument("--output", type=Path, default=Path("output/rates/quick"))
    parser.add_argument("--render-only", action="store_true")
    parser.add_argument("--workers", type=int, help="Worker processes for independent hedge paths.")
    options = parser.parse_args()
    cfg = yaml.safe_load(options.config.read_text(encoding="utf-8"))
    if options.workers is not None:
        cfg["workers"] = options.workers

    from quantproj.rates.reporting import render

    if not options.render_only:
        run(cfg, options.output)
    render(options.output, cfg)
    print(f"Rates report: {options.output / 'rates_validation.pdf'}", flush=True)


if __name__ == "__main__":
    main()
