import plotly.graph_objects as go
import numpy as np
import pandas as pd
import logging
from typing import Optional, Any, Dict, List # Added List
from zoneinfo import ZoneInfo # Assuming this might be used by SuiteConfig/EvalResult or _get_dataframe
import json
import os

logger = logging.getLogger(__name__)

INFORMAL_TO_FORMAL_NAME_MAP = {
    # Short Names
    "lit": "Literature Understanding",
    "data": "Data Analysis",
    "code": "Code Execution",
    "discovery": "Discovery",

    # Long Raw Names
    "arxivdigestables_validation": "Arxivdigestables Validation",
    "sqa_dev": "Sqa Dev",
    "litqa2_validation": "Litqa2 Validation",
    "paper_finder_validation": "Paper Finder Validation",
    "discoverybench_validation": "Discoverybench Validation",
    "core_bench_validation": "Core Bench Validation",
    "ds1000_validation": "DS1000 Validation",
    "e2e_discovery_validation": "E2E Discovery Validation",
    "super_validation": "Super Validation",
}

### 2. The Updated Helper Functions ###

def _safe_round(value, digits=2):
    """Rounds a number if it's a valid float/int, otherwise returns it as is."""
    return round(value, digits) if isinstance(value, (float, int)) and pd.notna(value) else value


def _pretty_column_name(raw_col: str) -> str:
    """
    Takes a raw column name from the DataFrame and returns a "pretty" version.
    Handles three cases:
    1. Fixed names (e.g., 'User/organization' -> 'Submitter').
    2. Dynamic names (e.g., 'ds1000_validation score' -> 'DS1000 Validation Score').
    3. Fallback for any other names.
    """
    # Case 1: Handle fixed, special-case mappings first.
    fixed_mappings = {
        'Agent': 'Agent',
        'Agent description': 'Agent Description',
        'User/organization': 'Submitter',
        'Submission date': 'Date',
        'Overall': 'Overall Score',
        'Overall cost': 'Overall Cost',
        'Logs': 'Logs'
    }
    if raw_col in fixed_mappings:
        return fixed_mappings[raw_col]

    # Case 2: Handle dynamic names by finding the longest matching base name.
    # We sort by length (desc) to match 'core_bench_validation' before 'core_bench'.
    sorted_base_names = sorted(INFORMAL_TO_FORMAL_NAME_MAP.keys(), key=len, reverse=True)

    for base_name in sorted_base_names:
        if raw_col.startswith(base_name):
            formal_name = INFORMAL_TO_FORMAL_NAME_MAP[base_name]

            # Get the metric part (e.g., ' score' or ' cost 95% CI')
            metric_part = raw_col[len(base_name):].strip()

            # Capitalize the metric part correctly (e.g., 'score' -> 'Score')
            pretty_metric = metric_part.capitalize()

            return f"{formal_name} {pretty_metric}"

    # Case 3: If no specific rule applies, just make it title case.
    return raw_col.title()


def create_pretty_tag_map(raw_tag_map: dict, name_map: dict) -> dict:
    """
    Converts a tag map with raw names into a tag map with pretty, formal names.

    Args:
        raw_tag_map: The map with raw keys and values (e.g., {'lit': ['litqa2_validation']}).
        name_map: The INFORMAL_TO_FORMAL_NAME_MAP used for translation.

    Returns:
        A new dictionary with pretty names (e.g., {'Literature Understanding': ['Litqa2 Validation']}).
    """
    pretty_map = {}
    # A reverse map to find raw keys from formal names if needed, though not used here
    # This is just for understanding; the main logic uses the forward map.

    # Helper to get pretty name with a fallback
    def get_pretty(raw_name):
        return name_map.get(raw_name, raw_name.replace("_", " ").title())

    for raw_key, raw_value_list in raw_tag_map.items():
        pretty_key = get_pretty(raw_key)
        pretty_value_list = [get_pretty(raw_val) for raw_val in raw_value_list]
        pretty_map[pretty_key] = sorted(list(set(pretty_value_list)))

    return pretty_map


def transform_raw_dataframe(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms a raw leaderboard DataFrame into a presentation-ready format.

    This function performs two main actions:
    1. Rounds all numeric metric values (columns containing 'score' or 'cost').
    2. Renames all columns to a "pretty", human-readable format.
    Args:
        raw_df (pd.DataFrame): The DataFrame with raw data and column names
                               like 'agent_name', 'overall/score', 'tag/code/cost'.
    Returns:
        pd.DataFrame: A new DataFrame ready for display.
    """
    if not isinstance(raw_df, pd.DataFrame):
        raise TypeError("Input 'raw_df' must be a pandas DataFrame.")

    df = raw_df.copy()

    # Create the mapping for pretty column names
    pretty_cols_map = {col: _pretty_column_name(col) for col in df.columns}

    # Rename the columns and return the new DataFrame
    transformed_df = df.rename(columns=pretty_cols_map)
    # Apply safe rounding to all metric columns
    for col in transformed_df.columns:
        if 'Score' in col or 'Cost' in col:
            transformed_df[col] = transformed_df[col].apply(_safe_round)

    logger.info("Raw DataFrame transformed: numbers rounded and columns renamed.")
    return transformed_df


class DataTransformer:
    """
    Visualizes a pre-processed leaderboard DataFrame.

    This class takes a "pretty" DataFrame and a tag map, and provides
    methods to view filtered versions of the data and generate plots.
    """
    def __init__(self, dataframe: pd.DataFrame, tag_map: dict[str, list[str]]):
        """
        Initializes the viewer.

        Args:
            dataframe (pd.DataFrame): The presentation-ready leaderboard data.
            tag_map (dict): A map of formal tag names to formal task names.
        """
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("Input 'dataframe' must be a pandas DataFrame.")
        if not isinstance(tag_map, dict):
            raise TypeError("Input 'tag_map' must be a dictionary.")

        self.data = dataframe
        self.tag_map = tag_map
        logger.info(f"DataTransformer initialized with a DataFrame of shape {self.data.shape}.")


    def view(
            self,
            tag: Optional[str] = "Overall", # Default to "Overall" for clarity
            use_plotly: bool = False,
    ) -> tuple[pd.DataFrame, dict[str, go.Figure]]:
        """
        Generates a filtered view of the DataFrame and a corresponding scatter plot.
        """
        if self.data.empty:
            logger.warning("No data available to view.")
            return self.data, {}

        # --- 1. Determine Primary and Group Metrics Based on the Tag ---
        # This logic is now explicit and easy to read.
        if tag is None or tag == "Overall":
            primary_metric = "Overall"
            # For the Overall view, the group IS all the main categories (the keys of the tag map).
            group_metrics = list(self.tag_map.keys())
        else:
            primary_metric = tag
            # For a specific tag, the group is its list of sub-tasks.
            group_metrics = self.tag_map.get(tag, [])

        # --- 2. Sort the DataFrame by the Primary Score ---
        primary_score_col = f"{primary_metric} Score"
        df_sorted = self.data
        if primary_score_col in self.data.columns:
            df_sorted = self.data.sort_values(primary_score_col, ascending=False, na_position='last')

        # --- 3. Build the List of Columns to Display ---
        base_cols = ["Agent", "Submitter", "Date", "Logs"]

        # Start with the primary metric score and cost
        metrics_to_display = [primary_score_col, f"{primary_metric} Cost"]

        # Add the score and cost for each item in our group
        for item in group_metrics:
            metrics_to_display.append(f"{item} Score")
            metrics_to_display.append(f"{item} Cost")

        # Combine base columns with metric columns, ensuring uniqueness and order
        final_cols_ordered = base_cols + list(dict.fromkeys(metrics_to_display))

        # Filter to only include columns that actually exist in our DataFrame
        existing_final_cols = [col for col in final_cols_ordered if col in df_sorted.columns]

        df_view = df_sorted[existing_final_cols].reset_index(drop=True)

        # --- 4. Generate the Scatter Plot for the Primary Metric ---
        plots: dict[str, go.Figure] = {}
        if use_plotly:
            primary_cost_col = f"{primary_metric} Cost"
            # Check if the primary score and cost columns exist in the FINAL view
            if primary_score_col in df_view.columns and primary_cost_col in df_view.columns:
                fig = _plot_scatter_plotly(
                    data=df_view,
                    x=primary_cost_col,
                    y=primary_score_col,
                    agent_col="Agent"
                )
                # Use a consistent key for easy retrieval later
                plots['scatter_plot'] = fig
            else:
                logger.warning(
                    f"Skipping plot for '{primary_metric}': score column '{primary_score_col}' "
                    f"or cost column '{primary_cost_col}' not found."
                )
                # Add an empty figure to avoid downstream errors
                plots['scatter_plot'] = go.Figure()

        return df_view, plots

DEFAULT_Y_COLUMN = "Overall Score"
DUMMY_X_VALUE_FOR_MISSING_COSTS = 0

def _plot_scatter_plotly(
        data: pd.DataFrame,
        x: Optional[str],
        y: str,
        agent_col: str = "Agent"
) -> go.Figure:

    # --- Steps 1-4: Data Validation and Preparation (No changes here) ---
    x_col_to_use = x
    y_col_to_use = y

    if y_col_to_use not in data.columns:
        logger.error(f"y-axis column '{y_col_to_use}' not found.")
        return go.Figure()
    if agent_col not in data.columns:
        logger.warning(f"Agent column '{agent_col}' not found.")
        return go.Figure()

    data_plot = data.copy()
    data_plot[y_col_to_use] = pd.to_numeric(data_plot[y_col_to_use], errors='coerce')

    x_axis_label = x if x else "Cost (Data N/A)"
    x_data_is_valid = False
    if x and x in data_plot.columns:
        try:
            data_plot[x_col_to_use] = pd.to_numeric(data_plot[x_col_to_use], errors='coerce')
            if data_plot[x_col_to_use].notna().any():
                x_data_is_valid = True
        except Exception as e:
            logger.warning(f"Error converting x-column '{x_col_to_use}' to numeric: {e}")

    if not x_data_is_valid:
        dummy_x_col_name = "__dummy_x_for_plotting__"
        data_plot[dummy_x_col_name] = DUMMY_X_VALUE_FOR_MISSING_COSTS
        x_col_to_use = dummy_x_col_name
        logger.info("Using dummy x-values for plotting.")

    # --- Step 5: Clean Data and Initialize Figure ---
    data_plot.dropna(subset=[y_col_to_use, x_col_to_use], inplace=True)
    fig = go.Figure()
    if data_plot.empty:
        logger.warning(f"No valid data to plot for y='{y_col_to_use}' and x='{x_col_to_use}'.")
        return fig

    # ========================================================================
    # --- NEW: Step 6 - Calculate and Draw the Efficiency Frontier Line ---
    # ========================================================================
    if x_data_is_valid:
        # Sort by cost (ascending), then by score (descending) to break ties
        sorted_data = data_plot.sort_values(by=[x_col_to_use, y_col_to_use], ascending=[True, False])

        frontier_points = []
        max_score_so_far = float('-inf')

        for index, row in sorted_data.iterrows():
            score = row[y_col_to_use]
            # If this point offers a better score than any we've seen before,
            # it's part of the frontier.
            if score > max_score_so_far:
                frontier_points.append({'x': row[x_col_to_use], 'y': score})
                max_score_so_far = score

        # Add the frontier line trace to the plot if we found any points
        if frontier_points:
            frontier_df = pd.DataFrame(frontier_points)
            fig.add_trace(go.Scatter(
                x=frontier_df['x'],
                y=frontier_df['y'],
                mode='lines',
                name='Efficiency Frontier',
                line=dict(color='firebrick', width=2, dash='dash'),
                hoverinfo='skip' # The line doesn't need a hover tooltip
            ))

    # --- Step 7: Plot Individual Agent Markers (No changes here) ---
    for agent, group in data_plot.groupby(agent_col):
        hover_x_display = "%{x:.2f}" if x_data_is_valid else "N/A"
        fig.add_trace(go.Scatter(
            x=group[x_col_to_use],
            y=group[y_col_to_use],
            mode='markers',
            name=str(agent),
            hovertemplate=f"{x_axis_label}: {hover_x_display}<br>{y_col_to_use}: %{{y:.2f}}<extra>{str(agent)}</extra>",
            marker=dict(size=10, opacity=0.8)
        ))

    # --- Step 8: Configure Layout (No changes here) ---
    xaxis_config = dict(title=x_axis_label)
    if not x_data_is_valid:
        xaxis_config['range'] = [DUMMY_X_VALUE_FOR_MISSING_COSTS - 1, DUMMY_X_VALUE_FOR_MISSING_COSTS + 1]
        xaxis_config['tickvals'] = [DUMMY_X_VALUE_FOR_MISSING_COSTS]
    else:
        xaxis_config['rangemode'] = "tozero"

    fig.update_layout(
        title=f"{y_col_to_use} vs. {x_axis_label}",
        xaxis=xaxis_config,
        yaxis=dict(title=y_col_to_use, rangemode="tozero"),
        legend_title_text=agent_col
    )

    return fig