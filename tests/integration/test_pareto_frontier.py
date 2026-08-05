import sys
from pathlib import Path

import pandas as pd

# The application uses a flat module layout rather than an installed package.
# Ensure those modules are importable under CI's `pytest tests/integration/` invocation.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import aliases
from leaderboard_transformer import _plot_scatter_plotly, get_pareto_df


def _tied_score_rows():
    return pd.DataFrame(
        [
            {
                "Agent": "cheaper-tie",
                "DS-1000 Score": 0.8533,
                "DS-1000 Cost": 0.0368,
                "Openness": aliases.CANONICAL_OPENNESS_OPEN_SOURCE_CLOSED_WEIGHTS,
                "Agent Tooling": aliases.CANONICAL_TOOL_USAGE_STANDARD,
                "Models Used": ["model-a"],
            },
            {
                "Agent": "dominated-tie",
                "DS-1000 Score": 0.8533,
                "DS-1000 Cost": 0.0519,
                "Openness": aliases.CANONICAL_OPENNESS_OPEN_SOURCE_CLOSED_WEIGHTS,
                "Agent Tooling": aliases.CANONICAL_TOOL_USAGE_STANDARD,
                "Models Used": ["model-b"],
            },
            {
                "Agent": "higher-score",
                "DS-1000 Score": 0.862,
                "DS-1000 Cost": 0.13,
                "Openness": aliases.CANONICAL_OPENNESS_OPEN_SOURCE_CLOSED_WEIGHTS,
                "Agent Tooling": aliases.CANONICAL_TOOL_USAGE_STANDARD,
                "Models Used": ["model-c"],
            },
        ]
    )


def test_tied_score_at_higher_cost_is_not_marked_pareto():
    pareto = get_pareto_df(_tied_score_rows())

    assert pareto["Agent"].tolist() == ["cheaper-tie", "higher-score"]


def test_tied_score_at_higher_cost_is_not_drawn_on_frontier():
    figure = _plot_scatter_plotly(
        _tied_score_rows(),
        x="DS-1000 Cost",
        y="DS-1000 Score",
        name="DS-1000",
    )

    frontier = next(trace for trace in figure.data if trace.name == "Efficiency Frontier")
    assert list(frontier.x) == [0.0368, 0.13]
    assert list(frontier.y) == [0.8533, 0.862]


def _equal_cost_and_score_rows():
    """Two entries equal on BOTH axes plus a same-cost lower-score entry.

    ``co-opt-a`` and ``co-opt-b`` are identical in cost and score, so neither
    dominates the other -> both belong on the frontier. ``same-cost-worse`` sits
    at the same cost but a lower score, so it is dominated and must be dropped.
    """
    def _row(agent, cost, score):
        return {
            "Agent": agent,
            "DS-1000 Score": score,
            "DS-1000 Cost": cost,
            "Openness": aliases.CANONICAL_OPENNESS_OPEN_SOURCE_CLOSED_WEIGHTS,
            "Agent Tooling": aliases.CANONICAL_TOOL_USAGE_STANDARD,
            "Models Used": ["m"],
        }

    return pd.DataFrame(
        [
            _row("co-opt-a", 0.0368, 0.8533),
            _row("co-opt-b", 0.0368, 0.8533),
            _row("same-cost-worse", 0.0368, 0.8000),
            _row("higher-score", 0.13, 0.862),
        ]
    )


def test_points_equal_on_both_axes_are_both_kept():
    pareto = get_pareto_df(_equal_cost_and_score_rows())

    assert sorted(pareto["Agent"].tolist()) == ["co-opt-a", "co-opt-b", "higher-score"]


def test_equal_cost_lower_score_is_dropped_on_frontier_line():
    figure = _plot_scatter_plotly(
        _equal_cost_and_score_rows(),
        x="DS-1000 Cost",
        y="DS-1000 Score",
        name="DS-1000",
    )

    frontier = next(trace for trace in figure.data if trace.name == "Efficiency Frontier")
    assert list(frontier.x) == [0.0368, 0.13]
    assert list(frontier.y) == [0.8533, 0.862]
