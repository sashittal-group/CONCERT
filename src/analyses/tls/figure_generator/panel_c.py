from __future__ import annotations

from .generate import DATA_ROOT, DEFAULT_OUTPUT_DIR
from .panel_sensitivity import (
    PANEL_C_FIGURE_SIZE,
    generate_boxplot_panel,
    load_profile_panel_data,
)


FIGURE_SIZE = PANEL_C_FIGURE_SIZE


def main() -> None:
    data = load_profile_panel_data(DATA_ROOT, DEFAULT_OUTPUT_DIR)
    for path in generate_boxplot_panel(
        data,
        DEFAULT_OUTPUT_DIR,
        figsize=FIGURE_SIZE,
    ).values():
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
