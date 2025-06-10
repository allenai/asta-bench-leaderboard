"""
View and plot leaderboard results.
"""

import logging
from typing import Optional
from zoneinfo import ZoneInfo

import datasets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from agenteval import compute_summary_statistics
from agenteval.config import SuiteConfig
from agenteval.models import EvalResult

logger = logging.getLogger(__name__)


class LeaderboardViewer:
    """
    Load and visualize leaderboard for a given HF dataset split.
    """

    def __init__(
            self, repo_id: str, config: str, split: str, is_internal: bool = False
    ):
        self._repo_id = repo_id
        self._config = config
        self._split = split
        self._internal = is_internal

        # build suite_config and mapping from tags to tasks from the first result
        # TODO: Verify the sort order
        ds = datasets.load_dataset(repo_id, name=config).get(split)
        if not ds:
            raise ValueError(f"Split '{split}' not found in dataset results")
        suite = EvalResult.model_validate(ds[0]).suite_config
        self._cfg = suite
        self.tag_map: dict[str, list[str]] = {}
        for task in suite.get_tasks(split):
            for t in task.tags or []:
                self.tag_map.setdefault(t, []).append(task.name)

    def _load(self):
        results = datasets.load_dataset(self._repo_id, name=self._config)
        overview = _get_dataframe(
            eval_results=results,
            split=self._split,
            is_internal=self._internal,
            suite_config=self._cfg,
        )
        return overview, self.tag_map

    def view(
            self, tag: Optional[str] = None, with_plots: bool = False
    ) -> tuple[pd.DataFrame, dict[str, plt.Figure]]:
        """
        If tag is None, primary="Overall" and group=all tags.
        Otherwise primary=tag and group=tasks under that tag.
        """
        data, tag_map = self._load()
        cols = [
            "Agent",
            "Submitter",
            "Completeness",
            "LLM Base",
            "Openness" ,
            "Date",
            "Logs",
        ]

        # choose primary metric and its sub‐group
        if tag is None:
            primary = "Overall"
            group = list(tag_map.keys())
        else:
            primary = tag
            group = tag_map.get(tag, [])
        data = data.sort_values(primary, ascending=False)

        # build full metric list: primary + its cost + each member and its cost
        metrics = [primary, f"{primary} cost"] + [
            m for t in group for m in (t, f"{t} cost")
        ]

        # filter to relevant columns
        ci_cols = [f"{m} 95% CI" for m in metrics if f"{m} 95% CI" in data.columns]
        df = data.loc[
             :,
             cols + [c for c in metrics if c in data.columns] + ci_cols,
             ].reset_index(drop=True)

        plots: dict[str, plt.Figure] = {}
        if with_plots:
            avail = [c for c in metrics if c in df.columns]
            for m in [primary] + group:
                x, y = f"{m} cost", m
                if x in df.columns and y in df.columns:
                    plots[f"scatter_{m}"] = _plot_scatter(
                        df, x=x, y=y, agent_col="Agent"
                    )

        return df, plots


def _get_dataframe(
        eval_results: datasets.DatasetDict,
        split: str,
        is_internal: bool,
        suite_config: SuiteConfig,
        timezone: str = "US/Pacific",
) -> pd.DataFrame:
    """
    Load leaderboard results from the given dataset split and return a DataFrame.
    """
    ds = eval_results.get(split)
    if not ds:
        cols = ["agent_name", "agent_description", "username", "submit_time"]
        pretty = [_pretty_column_name(c) for c in cols]
        empty = pd.DataFrame({c: ["No data"] for c in pretty})
        return empty

    cfg = suite_config

    rows = []
    for itm in ds:
        ev = EvalResult.model_validate(itm)
        sub = ev.submission
        # only format if submit_time present, else leave as None
        ts = sub.submit_time
        if ts is not None:
            date = ts.astimezone(ZoneInfo(timezone)).strftime("%Y-%m-%d")
        else:
            date = None

        if not ev.results:
            logger.warning(
                f"Skipping submission {sub.agent_name} ({sub.username}) "
                f"({sub.submit_time}) with no results"
            )
            continue
        stats = compute_summary_statistics(
            suite_config=cfg, split=split, results=ev.results
        )
        flat = {}
        for key, s in stats.items():
            parts = key.split("/")
            if parts[0] == "overall":
                flat["overall/score"], flat["overall/cost"] = s.score, s.cost
            elif parts[0] == "tag":
                flat[f"tag/{parts[1]}/score"], flat[f"tag/{parts[1]}/cost"] = (
                    s.score,
                    s.cost,
                )
            else:  # task
                t0 = parts[1]
                # compute 95% CI half-width from stderr
                flat.update(
                    {
                        f"task/{t0}/score": s.score,
                        f"task/{t0}/score_ci": (
                            (s.score_stderr * 1.96)
                            if s.score_stderr is not None
                            else np.nan
                        ),
                        f"task/{t0}/cost": s.cost,
                        f"task/{t0}/cost_ci": (
                            (s.cost_stderr * 1.96)
                            if s.cost_stderr is not None
                            else np.nan
                        ),
                    }
                )

        rows.append(
            {
                "agent_name": sub.agent_name,
                "username": sub.username or "",
                "submit_time": date,
                **flat,
                "logs_url": sub.logs_url if is_internal else sub.logs_url_public,
            }
        )

    df = pd.DataFrame(rows)

    # prepare pretty column mapping
    pretty_cols = {c: _pretty_column_name(c) for c in df.columns}

    # construct overview table with human-friendly names
    overview = df.rename(columns=pretty_cols)

    return overview


def _pretty_column_name(col: str) -> str:
    """Map raw column name to display name."""
    # fixed mappings
    mapping = {
        "submit_time": "Date",
        "agent_name": "Agent",
        "username": "User/organization",
        "logs_url": "Logs",
        "overall/score": "Score",
        "overall/cost": "Overall Cost (USD)",
    }
    if col in mapping:
        return mapping[col]
    # dynamic: task/{name}/{metric} or tag/{name}/{metric}
    parts = col.split("/")
    if len(parts) == 3:
        _, name, metric = parts
        if metric == "score":
            return name
        if metric == "cost":
            return f"{name} cost"
        if metric == "score_ci":
            return f"{name} 95% CI"
        if metric == "cost_ci":
            return f"{name} cost 95% CI"
    # fallback to last segment
    return parts[-1]


# def _plot_hbar(
#         data: pd.DataFrame,
#         agent_col: str,
#         metrics: list[str],
# ) -> plt.Figure:
#     """Horizontal bar chart of metrics, one row per agent."""
#     n = len(metrics)
#     # color each metric pair the same
#     group_count = (n + 1) // 2
#     palette = sns.color_palette(n_colors=group_count)
#
#     # Set minimum width per subplot for readable x-axis labels, let height auto-size
#     min_width_per_subplot = 4
#     fig_width = n * min_width_per_subplot
#     fig, axes = plt.subplots(
#         ncols=n, sharey=True, figsize=(fig_width, plt.rcParams["figure.figsize"][1])
#     )
#
#     if n == 1:
#         axes = [axes]
#
#     for idx, (ax, metric) in enumerate(zip(axes, metrics)):
#         color = palette[idx // 2]
#
#         sns.barplot(data=data, y=agent_col, x=metric, ax=ax, color=color)
#         ci = data.get(f"{metric} 95% CI")
#         if ci is not None:
#             y_positions = range(len(data))
#             ax.errorbar(
#                 x=data[metric],
#                 y=y_positions,
#                 xerr=ci,
#                 fmt="none",
#                 ecolor="gray",
#                 capsize=3,
#             )
#         ax.set_xlabel(metric)
#         ax.set_xlim(left=0)
#
#     plt.tight_layout()
#     return fig


def _plot_scatter(
        data: pd.DataFrame,
        x: str,  # Cost column name (e.g., "Overall cost")
        y: str,  # Score column name (e.g., "Overall score")
        agent_col: str,
) -> plt.Figure:
    """Scatter plot of agent results, showing score vs cost with Pareto frontier."""
    fig, ax = plt.subplots(figsize=(20,7))

    # Make a copy for manipulation to find frontier without affecting original data
    plot_data = data.copy()

    # Ensure score (y) and cost (x) are numeric and drop NaNs for frontier calculation
    plot_data[y] = pd.to_numeric(plot_data[y], errors='coerce')
    plot_data[x] = pd.to_numeric(plot_data[x], errors='coerce')
    frontier_data = plot_data.dropna(subset=[y, x])

    if not frontier_data.empty:
        # Sort by cost (x) ascending, then by score (y) descending for tie-breaking
        frontier_data = frontier_data.sort_values(by=[x, y], ascending=[True, False])

        pareto_points = []
        max_score_at_cost = -np.inf  # Initialize with negative infinity

        for index, row in frontier_data.iterrows():
            current_score = row[y]
            current_cost = row[x]
            # Only add point if it offers a higher score than any previous point
            # on the frontier with less or equal cost (implicit by sorting).
            # More strictly, for a point to be on the frontier here, it must improve the score.
            if current_score > max_score_at_cost:
                # Optional: If allowing same score but lower cost (already handled by sort somewhat)
                # you might need to check if a point with same score but lower cost exists
                # For this algorithm, we simply take points that strictly increase score.
                pareto_points.append(row)
                max_score_at_cost = current_score

        if pareto_points:
            pareto_df = pd.DataFrame(pareto_points)
            # Sort pareto_df by cost again just to be sure for plotting line
            pareto_df = pareto_df.sort_values(by=x)
            # Plot the Pareto frontier line
            ax.plot(pareto_df[x], pareto_df[y], marker='o', linestyle='-', color='red', alpha=0.7, linewidth=2, markersize=5, label='Pareto Frontier')

    # Plot all data points
    sns.scatterplot(data=data, x=x, y=y, hue=agent_col, s=100, ax=ax, legend="auto")

    # Error bars (if they exist)
    x_ci_col = f"{x} 95% CI"
    y_ci_col = f"{y} 95% CI"
    if x_ci_col in data.columns or y_ci_col in data.columns:
        # Filter data for error bars to only include rows present in the original 'data'
        # This is important if 'frontier_data' subset was used for some logic but error bars are for all.
        error_bar_data = data.copy() # Use original data for error bars
        error_bar_data[x_ci_col] = pd.to_numeric(error_bar_data.get(x_ci_col), errors='coerce')
        error_bar_data[y_ci_col] = pd.to_numeric(error_bar_data.get(y_ci_col), errors='coerce')

        ax.errorbar(
            x=error_bar_data[x], # Use original data's x
            y=error_bar_data[y], # Use original data's y
            xerr=error_bar_data.get(x_ci_col),
            yerr=error_bar_data.get(y_ci_col),
            fmt="none",
            ecolor="gray",
            alpha=0.5,
            capsize=3,
            zorder=0 # Draw error bars behind scatter points
        )

    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0) # Scores and costs are typically non-negative
    ax.set_xlabel(x) # x is cost
    ax.set_ylabel(y) # y is score

    # Adjust legend: Get handles and labels from seaborn plot, then add frontier's
    handles, labels = ax.get_legend_handles_labels()
    # Check if "Pareto Frontier" was actually plotted and add its handle/label if so
    if pareto_points and "Pareto Frontier" not in labels: # Avoid duplicate legend items
        # Find the frontier line object to get its handle
        frontier_line = next((line for line in ax.get_lines() if line.get_label() == 'Pareto Frontier'), None)
        if frontier_line:
            handles.append(frontier_line)
            labels.append('Pareto Frontier')

    ax.legend(handles=handles, labels=labels, title=agent_col, bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)

    plt.tight_layout(rect=[0, 0, 0.85, 1])
    return fig


__all__ = ["LeaderboardViewer"]