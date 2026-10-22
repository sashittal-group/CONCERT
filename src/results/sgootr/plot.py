from __future__ import annotations

import argparse
from pathlib import Path

from src.results.tls.config import load_config
from src.results.tls.plot import plot_sensitivity


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot SGOOTR crown-conditioned profile intervals."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().with_name("config.yaml"),
    )
    args = parser.parse_args()
    config = load_config(args.config, section="sgootr_profiles")
    for output in plot_sensitivity(config):
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()
