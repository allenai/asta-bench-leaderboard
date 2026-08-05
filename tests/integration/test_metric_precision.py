import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import aliases
from leaderboard_transformer import (
    _plot_scatter_plotly,
    format_cost_column,
    format_score_column,
)


def test_table_metrics_show_three_decimals_and_full_precision_on_hover():
    frame = pd.DataFrame(
        {
            "Score": [0.8533333333333334],
            "Score CI": [0.023126041021506245],
            "Cost": [0.036786123456789],
            "Cost CI": [0.001995123456789],
        }
    )

    rendered = format_cost_column(frame.copy(), "Cost", "Cost CI")
    rendered = format_score_column(rendered, "Score", "Score CI")

    assert ">$0.037</span>" in rendered.loc[0, "Cost"]
    assert 'data-tooltip="Cost (USD): 0.0367861234568' in rendered.loc[0, "Cost"]
    assert "Cost (USD): 0.0367861234568" in rendered.loc[0, "Cost"]
    assert "95% CI: ±0.00199512345679" in rendered.loc[0, "Cost"]
    assert ">0.853</span>" in rendered.loc[0, "Score"]
    assert "Score: 0.853333333333" in rendered.loc[0, "Score"]


def test_benchmark_plot_exposes_precision_and_confidence_intervals():
    frame = pd.DataFrame(
        {
            "Agent": ["RoboPhD"],
            "Openness": [aliases.CANONICAL_OPENNESS_OPEN_SOURCE_OPEN_WEIGHTS],
            "Agent Tooling": [aliases.CANONICAL_TOOL_USAGE_STANDARD],
            "Models Used": [["model"]],
            "Score": [0.8533333333333334],
            "Score CI": [0.023126041021506245],
            "Cost": [0.036786123456789],
            "Cost CI": [0.001995123456789],
        }
    )

    figure = _plot_scatter_plotly(
        frame,
        x="Cost",
        y="Score",
        x_ci="Cost CI",
        y_ci="Score CI",
    )
    marker_trace = next(trace for trace in figure.data if trace.mode == "markers")

    assert list(marker_trace.error_x.array) == [0.001995123456789]
    assert list(marker_trace.error_y.array) == [0.023126041021506245]
    assert "Score: <b>0.853333333333</b>" in marker_trace.text[0]
    assert "Cost 95% CI: <b>±$0.00199512345679</b>" in marker_trace.text[0]


def test_plot_without_confidence_intervals_keeps_error_bars_hidden():
    frame = pd.DataFrame(
        {
            "Agent": ["agent"],
            "Openness": [aliases.CANONICAL_OPENNESS_OPEN_SOURCE_OPEN_WEIGHTS],
            "Agent Tooling": [aliases.CANONICAL_TOOL_USAGE_STANDARD],
            "Models Used": [["model"]],
            "Score": [0.5],
            "Cost": [1.25],
        }
    )

    figure = _plot_scatter_plotly(frame, x="Cost", y="Score")
    marker_trace = next(trace for trace in figure.data if trace.mode == "markers")

    assert marker_trace.error_x.visible is False
    assert marker_trace.error_y.visible is False
