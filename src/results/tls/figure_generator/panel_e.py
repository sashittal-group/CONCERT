from __future__ import annotations

from .generate import DATA_ROOT, DEFAULT_OUTPUT_DIR
from .panel_sensitivity import load_profile_panel_data, generate_profile_panel


FIGURE_SIZE = (5.2, 4.0)


def main() -> None:
    data = load_profile_panel_data(DATA_ROOT, DEFAULT_OUTPUT_DIR)
    for path in generate_profile_panel(
        "E",
        data,
        DEFAULT_OUTPUT_DIR,
        figsize=FIGURE_SIZE,
    ).values():
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
