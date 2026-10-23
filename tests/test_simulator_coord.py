from __future__ import annotations

import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import networkx as nx
import numpy as np
import pandas as pd

from src.simulator_coord import (
    _apply_scheduled_transitions,
    simulate_multitype_bd_with_diff,
    simulate_from_settings,
)


class SimulatorCoordTests(unittest.TestCase):
    def test_simulation_applies_transition_at_scheduled_time(self) -> None:
        differentiation_times = nx.DiGraph()
        differentiation_times.add_node("A", transition_times=[0.5])
        differentiation_times.add_node("B", transition_times=[])
        differentiation_times.add_edge("A", "B")
        transition_probabilities = pd.DataFrame(
            [[0.0, 1.0], [0.0, 1.0]],
            index=["A", "B"],
            columns=["A", "B"],
        )

        graph, root = simulate_multitype_bd_with_diff(
            birth_rates={"A": 0.0, "B": 0.0},
            death_rates={"A": 0.0, "B": 0.0},
            sampling_rates={"A": 1.0, "B": 1.0},
            transition_probabilities=transition_probabilities,
            differentiation_times=differentiation_times,
            time_length=1.0,
            seed=7,
        )

        transitions = [
            node
            for node, data in graph.nodes(data=True)
            if data["event"] == "transition"
        ]
        samples = [
            data["state"]
            for _, data in graph.nodes(data=True)
            if data["event"] == "sample"
        ]
        self.assertEqual(len(transitions), 1)
        self.assertEqual(graph.nodes[transitions[0]]["time"], 0.5)
        self.assertEqual(samples, ["B"])
        self.assertEqual(graph.nodes[root]["state"], "A")

    def test_only_lineages_with_a_scheduled_state_transition_are_updated(self) -> None:
        graph = nx.DiGraph()
        graph.add_node(0, time=0.0, event="root", state="A")
        graph.add_node(1, time=None, event=None, state="A")
        graph.add_node(2, time=None, event=None, state="B")
        graph.add_edge(
            0,
            1,
            start_time=0.0,
            end_time=None,
            length=None,
        )
        graph.add_edge(
            0,
            2,
            start_time=0.0,
            end_time=None,
            length=None,
        )

        differentiation_times = nx.DiGraph()
        differentiation_times.add_node("A", transition_times=[1.0])
        differentiation_times.add_node("B", transition_times=[2.0])
        differentiation_times.add_edge("A", "B")
        transition_probabilities = pd.DataFrame(
            [[0.0, 1.0], [0.0, 1.0]],
            index=["A", "B"],
            columns=["A", "B"],
        )
        alive = [(0, 1, "A"), (0, 2, "B")]

        updated_alive, next_node = _apply_scheduled_transitions(
            graph=graph,
            alive=alive,
            differentiation_times=differentiation_times,
            transition_probabilities=transition_probabilities,
            transition_time=1.0,
            next_node=3,
            rng=np.random.default_rng(1),
        )

        self.assertEqual(graph.nodes[1]["event"], "transition")
        self.assertEqual(graph.edges[0, 1]["end_time"], 1.0)
        self.assertIsNone(graph.nodes[2]["event"])
        self.assertIsNone(graph.edges[0, 2]["end_time"])
        self.assertEqual(updated_alive, [(1, 3, "B"), (0, 2, "B")])
        self.assertEqual(next_node, 4)

    def test_scheduled_transition_accepts_read_only_probabilities(self) -> None:
        graph = nx.DiGraph()
        graph.add_node(0, time=0.0, event="root", state="A")
        graph.add_node(1, time=None, event=None, state="A")
        graph.add_edge(
            0,
            1,
            start_time=0.0,
            end_time=None,
            length=None,
        )

        differentiation_times = nx.DiGraph()
        differentiation_times.add_node("A", transition_times=[1.0])
        differentiation_times.add_node("B", transition_times=[])
        differentiation_times.add_edge("A", "B")
        probability_data = np.array([[0.0, 1.0], [0.0, 1.0]])
        probability_data.setflags(write=False)
        transition_probabilities = pd.DataFrame(
            probability_data,
            index=["A", "B"],
            columns=["A", "B"],
            copy=False,
        )

        updated_alive, next_node = _apply_scheduled_transitions(
            graph=graph,
            alive=[(0, 1, "A")],
            differentiation_times=differentiation_times,
            transition_probabilities=transition_probabilities,
            transition_time=1.0,
            next_node=2,
            rng=np.random.default_rng(1),
        )

        self.assertEqual(updated_alive, [(1, 2, "B")])
        self.assertEqual(next_node, 3)

    def test_seeded_simulation_does_not_change_global_random_state(self) -> None:
        differentiation_times = nx.DiGraph()
        differentiation_times.add_node("A", transition_times=[])
        transition_probabilities = pd.DataFrame(
            [[1.0]],
            index=["A"],
            columns=["A"],
        )
        python_state = random.getstate()
        numpy_state = np.random.get_state()

        simulate_multitype_bd_with_diff(
            birth_rates={"A": 0.0},
            death_rates={"A": 0.0},
            sampling_rates={"A": 1.0},
            transition_probabilities=transition_probabilities,
            differentiation_times=differentiation_times,
            time_length=1.0,
            seed=7,
        )

        self.assertEqual(random.getstate(), python_state)
        current_numpy_state = np.random.get_state()
        self.assertEqual(current_numpy_state[0], numpy_state[0])
        np.testing.assert_array_equal(current_numpy_state[1], numpy_state[1])
        self.assertEqual(current_numpy_state[2:], numpy_state[2:])

    def test_simulate_from_settings_stops_after_max_attempts(self) -> None:
        setting_text = """\
# num states
1
# states
A
# birth rates
A 0
# death rates
A 0
# sampling rates
A 0
# transition times
A -
# transition probabilities
A -> A 1
# simulation time horizon
1
# initial state
A
# num trees
1
"""
        empty_sample_graph = nx.DiGraph()
        empty_sample_graph.add_node(0, event="unsampled", state="A")

        with tempfile.TemporaryDirectory() as temporary_directory:
            setting_path = Path(temporary_directory) / "setting.txt"
            setting_path.write_text(setting_text)
            with patch(
                "src.simulator_coord.simulate_multitype_bd_with_diff",
                return_value=(empty_sample_graph, 0),
            ) as simulate:
                with self.assertRaisesRegex(RuntimeError, "after 2 attempts"):
                    simulate_from_settings(
                        setting_file=setting_path,
                        source_seed=17,
                        output_dir=Path(temporary_directory) / "output",
                        max_attempts=2,
                    )

            self.assertEqual(simulate.call_count, 2)


if __name__ == "__main__":
    unittest.main()
