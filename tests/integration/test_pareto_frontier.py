import pandas as pd

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
