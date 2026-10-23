from __future__ import annotations

from src.main2 import main as run_inference


def main() -> None:
    run_inference(require_initial_parameters=True)


if __name__ == "__main__":
    main()
