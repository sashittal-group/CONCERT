from __future__ import annotations

import argparse
from collections import Counter
import csv
from pathlib import Path
import pickle
from typing import Any, Mapping, Sequence

from src.results.tls.config import PipelineConfig, load_config


def build_leaf_count_rows(
    trees_by_dataset: Mapping[str, Sequence[Any]],
    datasets: Sequence[str],
    model_states: Sequence[str],
) -> tuple[list[dict[str, int | str]], list[str]]:
    counts_by_dataset: dict[str, list[tuple[Counter[str], int]]] = {}
    all_states = set(model_states)

    for dataset in datasets:
        trees = trees_by_dataset.get(dataset)
        if not trees:
            raise ValueError(f"No trees provided for dataset {dataset}")

        tree_counts = []
        for tree_index, tree in enumerate(trees, start=1):
            if not callable(getattr(tree, "iter_leaves", None)):
                raise TypeError(f"Tree {tree_index} in {dataset} has no iter_leaves()")
            counts: Counter[str] = Counter()
            unlabeled_leaf_count = 0
            for leaf in tree.iter_leaves():
                state = getattr(leaf, "state", None)
                if state is None or (isinstance(state, str) and not state.strip()):
                    unlabeled_leaf_count += 1
                elif not isinstance(state, str):
                    raise TypeError(
                        f"Leaf state in {dataset} tree_{tree_index:03d} "
                        f"must be a string or None; got {type(state).__name__}"
                    )
                else:
                    counts[state] += 1
            all_states.update(counts)
            tree_counts.append((counts, unlabeled_leaf_count))
        counts_by_dataset[dataset] = tree_counts

    states = [*model_states, *sorted(all_states.difference(model_states))]
    rows: list[dict[str, int | str]] = []
    for dataset in datasets:
        dataset_totals: Counter[str] = Counter()
        unlabeled_total = 0
        for tree_index, (counts, unlabeled_leaf_count) in enumerate(
            counts_by_dataset[dataset],
            start=1,
        ):
            dataset_totals.update(counts)
            unlabeled_total += unlabeled_leaf_count
            rows.append(
                {
                    "dataset": dataset,
                    "tree_name": f"tree_{tree_index:03d}",
                    **{state: counts[state] for state in states},
                    "unlabeled_leaf_count": unlabeled_leaf_count,
                    "total_labeled_leaves": sum(counts.values()),
                }
            )
        rows.append(
            {
                "dataset": dataset,
                "tree_name": "TOTAL",
                **{state: dataset_totals[state] for state in states},
                "unlabeled_leaf_count": unlabeled_total,
                "total_labeled_leaves": sum(dataset_totals.values()),
            }
        )
    return rows, states


def write_leaf_count_table(config: PipelineConfig) -> Path:
    trees_by_dataset = {}
    for dataset in config.datasets:
        tree_path = config.dataset_root(dataset) / "processed_data" / "trees.pkl"
        with tree_path.open("rb") as tree_file:
            trees_by_dataset[dataset] = pickle.load(tree_file)

    rows, states = build_leaf_count_rows(
        trees_by_dataset,
        config.datasets,
        config.state_names,
    )
    output = (
        config.preprocessing_root
        / config.plotting["leaf_state_counts_filename"]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "tree_name",
        *states,
        "unlabeled_leaf_count",
        "total_labeled_leaves",
    ]
    with output.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote per-tree labeled leaf counts to {output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Count labeled leaves by tree and cell state for TLS/TLSC."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().with_name("config.yaml"),
    )
    args = parser.parse_args()
    write_leaf_count_table(load_config(args.config))


if __name__ == "__main__":
    main()
