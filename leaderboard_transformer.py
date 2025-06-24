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
            tag: Optional[str] = None,
            use_plotly: bool = False,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Generates a filtered view of the leaderboard DataFrame."""
        if self.data.empty:
            logger.warning("No data available to view.")
            return self.data, {}

        base_cols = ["Agent", "Submitter", "Date", "Logs"]
        existing_cols = [col for col in base_cols if col in self.data.columns]

        primary = tag if tag else "Overall"
        group = self.tag_map.get(tag, list(self.tag_map.keys()) if tag is None else [])

        df_sorted = self.data
        if f"{primary} Score" in self.data.columns:
            df_sorted = self.data.sort_values(f"{primary} Score", ascending=False)

        metrics_to_display = []
        print(f"I'M A PRIMARY COLOR BITCH {primary} IM THE GROUP{group}")
        if f"{primary} Score" in df_sorted.columns: metrics_to_display.append(f"{primary} Score")
        if f"{primary} Cost" in df_sorted.columns: metrics_to_display.append(f"{primary} Cost")

        for g_item in group:
            if f"{g_item} Score" in df_sorted.columns: metrics_to_display.append(f"{g_item} Score")
            if f"{g_item} Cost" in df_sorted.columns: metrics_to_display.append(f"{g_item} Cost")

        final_cols = existing_cols + list(dict.fromkeys(metrics_to_display))
        df_view = df_sorted.loc[:, [c for c in final_cols if c in df_sorted.columns]].reset_index(drop=True)

        plots: dict[str, Any] = {}
        if use_plotly:
            plot_metric_names = [primary] + [g_item for g_item in group if g_item in df_sorted.columns]
            for metric_name in plot_metric_names:
                score_col = f"{metric_name} Score"
                cost_col = f"{metric_name} Cost"
                if score_col in df_view.columns and cost_col in df_view.columns:
                    if use_plotly:
                        fig = _plot_scatter_plotly(df_view, x=cost_col, y=score_col, agent_col="Agent")
                    plots[f"scatter_{metric_name}"] = fig
                else:
                    logger.warning(
                        f"Skipping plot for '{metric_name}': score column '{score_col}' or cost column '{cost_col}' not found."
                    )

        return df_view, plots

DEFAULT_Y_COLUMN = "Overall Score"
DUMMY_X_VALUE_FOR_MISSING_COSTS = 0 # Value to use if x-axis data (costs) is missing

def _plot_scatter_plotly(
        data: pd.DataFrame,
        x: Optional[str],
        y: str,
        agent_col: str = "Agent"
) -> go.Figure:

    x_col_to_use = x
    y_col_to_use = y

    # 1. Check if y-column exists
    if y_col_to_use not in data.columns:
        logger.error(
            f"y-axis column '{y_col_to_use}' MUST exist in DataFrame. "
            f"Cannot generate plot. Available columns: {data.columns.tolist()}"
        )
        return go.Figure()

    # 2. Check if agent_col exists
    if agent_col not in data.columns:
        logger.warning(
            f"Agent column '{agent_col}' not found in DataFrame. "
            f"Available columns: {data.columns.tolist()}. Returning empty figure."
        )
        return go.Figure()

    # 3. Prepare data (make a copy, handle numeric conversion for y)
    data_plot = data.copy()
    try:
        data_plot[y_col_to_use] = pd.to_numeric(data_plot[y_col_to_use], errors='coerce')
    except Exception as e:
        logger.error(f"Error converting y-column '{y_col_to_use}' to numeric: {e}. Returning empty figure.")
        return go.Figure()

    # 4. Handle x-column (costs)
    x_axis_label = x_col_to_use if x_col_to_use else "Cost (Data N/A)" # Label for the x-axis
    x_data_is_valid = False

    if x_col_to_use and x_col_to_use in data_plot.columns:
        try:
            data_plot[x_col_to_use] = pd.to_numeric(data_plot[x_col_to_use], errors='coerce')
            # Check if there's any non-NaN data after coercion for x
            if data_plot[x_col_to_use].notna().any():
                x_data_is_valid = True
            else:
                logger.info(f"x-axis column '{x_col_to_use}' exists but contains all NaN/None values after numeric conversion.")
        except Exception as e:
            logger.warning(f"Error converting x-column '{x_col_to_use}' to numeric: {e}. Will use dummy x-values.")
            # x_data_is_valid remains False
    else:
        if x_col_to_use: # Name was provided but column doesn't exist
            logger.warning(f"x-axis column '{x_col_to_use}' not found in DataFrame.")
        else: # x (column name) was None
            logger.info("x-axis column name was not provided (is None).")

    if not x_data_is_valid:
        logger.info(f"Using dummy x-value '{DUMMY_X_VALUE_FOR_MISSING_COSTS}' for all data points as x-data is missing or invalid.")
        # Create a new column with the dummy x-value for all rows
        # Use a unique name for this dummy column to avoid potential clashes
        dummy_x_col_name = "__dummy_x_for_plotting__"
        data_plot[dummy_x_col_name] = DUMMY_X_VALUE_FOR_MISSING_COSTS
        x_col_to_use = dummy_x_col_name # Update x_col_to_use to point to our dummy data
        x_axis_label = x if x else "Cost (Data N/A)" # Use original x name for label if provided
        # or a generic label if x was None.
        # Could also be f"Cost (Fixed at {DUMMY_X_VALUE_FOR_MISSING_COSTS})"


    # 5. Drop rows where y is NaN (x is now guaranteed to have values, either real or dummy)
    data_plot.dropna(subset=[y_col_to_use], inplace=True)

    fig = go.Figure()

    if data_plot.empty:
        logger.warning(f"No valid data to plot for y='{y_col_to_use}' (and x='{x_col_to_use}') after cleaning NaNs from y.")
        # Still return a figure object, but it will be empty. Update layout for clarity.
        fig.update_layout(
            title=f"{y_col_to_use} vs. {x_axis_label} (No Data)",
            xaxis=dict(title=x_axis_label, range=[DUMMY_X_VALUE_FOR_MISSING_COSTS - 1, DUMMY_X_VALUE_FOR_MISSING_COSTS + 1] if not x_data_is_valid else None),
            yaxis=dict(title=y_col_to_use)
        )
        return fig


    for agent, group in data_plot.groupby(agent_col):
        hover_x_display = "%{x:.2f}" if x_data_is_valid else str(DUMMY_X_VALUE_FOR_MISSING_COSTS) + " (fixed)"
        fig.add_trace(go.Scatter(
            x=group[x_col_to_use],
            y=group[y_col_to_use],
            mode='markers',
            name=str(agent),
            hovertemplate=f"{x_axis_label}: {hover_x_display}<br>{y_col_to_use}: %{{y:.2f}}<extra>{str(agent)}</extra>",
            marker=dict(size=10)
        ))

    # Configure layout
    xaxis_config = dict(title=x_axis_label)
    if not x_data_is_valid: # If using dummy x, set a tighter, fixed range for x-axis
        xaxis_config['range'] = [DUMMY_X_VALUE_FOR_MISSING_COSTS - 1, DUMMY_X_VALUE_FOR_MISSING_COSTS + 1]
        xaxis_config['tickvals'] = [DUMMY_X_VALUE_FOR_MISSING_COSTS] # Show only one tick at the dummy value
        xaxis_config['ticktext'] = [str(DUMMY_X_VALUE_FOR_MISSING_COSTS)]
    else: # Real x-data
        xaxis_config['rangemode'] = "tozero"


    fig.update_layout(
        title=f"{y_col_to_use} vs. {x_axis_label}",
        xaxis=xaxis_config,
        yaxis=dict(title=y_col_to_use, rangemode="tozero"),
        legend_title_text=agent_col
    )

    return fig