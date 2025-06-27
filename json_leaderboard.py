import logging
from typing import Optional, Any, Dict # Added Dict
from zoneinfo import ZoneInfo

# datasets import might not be strictly needed by LeaderboardViewer itself anymore,
# but _get_dataframe might still use types from it if EvalResult refers to them.
# For now, let's keep it if your EvalResult or SuiteConfig models have dependencies.
# If not, it can be removed from here.
import datasets # Potentially removable from this file
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import seaborn as sns
import json # For loading the local JSON file
import os # For checking file existence

from agenteval import compute_summary_statistics
from agenteval.config import SuiteConfig
from agenteval.models import EvalResult

logger = logging.getLogger(__name__)

import logging
from typing import Optional, Any, Dict, List # Added List
from zoneinfo import ZoneInfo # Assuming this might be used by SuiteConfig/EvalResult or _get_dataframe
import json
import os

# Assuming these are correctly imported from your project
from agenteval.config import SuiteConfig
from agenteval.models import EvalResult
# from agenteval import compute_summary_statistics # Used by _get_dataframe


class DataTransformer:
    """
    Load and visualize leaderboard from a single, local JSON result file.
    """
    _INFORMAL_TO_FORMAL_NAME_MAP = {
        "lit": "Literature Understanding",
        "data": "Data Analysis",
        "code": "Code Execution",
        "discovery": "Discovery",
        "arxivdigestables_validation": "Arxivdigestables Validation",
        "sqa_dev": "Sqa Dev",
        "litqa2_validation": "Litqa2 Validation",
        "paper_finder_validation": "Paper Finder Validation",
        "discoverybench_validation": "Discoverybench Validation",
        "core_bench_validation": "Core Bench Validation",
        "ds1000_validation": "DS1000 Validation",
        "e2e_discovery_validation": "E2E Discovery Validation",
        "super_validation": "Super Validation",
        # Add any other raw names that can appear in task.name or task.tags
    }

    def __init__(
            self,
            json_file_path: str, # Mandatory: path to the local JSON file
            split: str,          # Still needed for context within the JSON's suite_config
            is_internal: bool = False
    ):
        self._json_file_path = json_file_path
        self._split = split
        self._internal = is_internal
        self._loaded_json_data: Optional[Dict[str, Any]] = None
        self._cfg: Optional[SuiteConfig] = None

        logger.info(f"Initializing LeaderboardViewer with local JSON file: {self._json_file_path}")

        # --- Load and Validate JSON data ---
        if not os.path.exists(self._json_file_path):
            raise FileNotFoundError(f"JSON file not found at path: {self._json_file_path}")
        try:
            with open(self._json_file_path, 'r', encoding='utf-8') as f:
                self._loaded_json_data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON from local file {self._json_file_path}: {e}")
        except Exception as e:
            raise ValueError(f"Error reading local file {self._json_file_path}: {e}")

        if not self._loaded_json_data:
            raise ValueError(f"No data loaded from JSON file {self._json_file_path}.")

        try:
            eval_result = EvalResult.model_validate(self._loaded_json_data)
        except Exception as e:
            raise ValueError(f"Failed to validate JSON data from file '{self._json_file_path}' against EvalResult model: {e}")

        self._cfg = eval_result.suite_config
        if not isinstance(self._cfg, SuiteConfig):
            raise TypeError(f"self._cfg is not a SuiteConfig object after loading from '{self._json_file_path}', got {type(self._cfg)}.")

        # --- Populate Tag Map (Corrected Placement and Helper Function Access) ---
        self.tag_map: dict[str, list[str]] = {}

        # Access tasks from the loaded config
        tasks_for_split: List[Any] = self._cfg.get_tasks(self._split) # Assuming get_tasks returns a list of task-like objects

        for task in tasks_for_split:
            # Ensure task object has 'name' and 'tags' attributes
            if not hasattr(task, 'name') or not hasattr(task, 'tags'):
                logger.warning(f"Task object {task} is missing 'name' or 'tags' attribute. Skipping.")
                continue

            formal_task_display_name = self._get_formal_display_name_static(task.name) # Use the helper method

            if not (task.tags or []):
                continue

            for raw_tag_name in task.tags:
                formal_tag_display_name_key = self._get_formal_display_name_static(raw_tag_name)

                self.tag_map.setdefault(formal_tag_display_name_key, []).append(formal_task_display_name)

        for key in self.tag_map:
            self.tag_map[key] = sorted(list(set(self.tag_map[key])))

    # --- Helper function defined as a static method or regular method ---
    # Option 1: Static method (doesn't need 'self', uses the class attribute)
    @staticmethod
    def _get_formal_display_name_static(raw_name: str) -> str:
        """
        Helper function to get the formal display name for a raw tag or task name.
        Uses the class's map and provides a fallback.
        """
        return DataTransformer._INFORMAL_TO_FORMAL_NAME_MAP.get(raw_name, raw_name.replace("_", " ").title())

    def _load(self) -> tuple[pd.DataFrame, dict[str, list[str]]]:
        """
        Prepares the DataFrame from the loaded JSON data.
        The JSON data is already loaded and validated in __init__.
        """
        if self._loaded_json_data is None or self._cfg is None:
            # This should not happen if __init__ completed successfully
            raise RuntimeError("LeaderboardViewer2 not properly initialized. JSON data or SuiteConfig is missing.")

        # The _get_dataframe function expects a list of records.
        # Since we have a single JSON file representing one result, wrap it in a list.
        records_list: list[dict] = [self._loaded_json_data]

        overview_df = _get_dataframe(
            records_list=records_list,
            split=self._split,
            is_internal=self._internal,
            suite_config=self._cfg, # Pass the SuiteConfig loaded in __init__
        )
        return overview_df, self.tag_map

    # --- view method remains the same as your last version ---
    def view(
            self,
            tag: Optional[str] = None,
            with_plots: bool = False,
            use_plotly: bool = False,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        data, tag_map = self._load() # tag_map is also returned by _load now
        print(f"AHAHASHJDBFGASJHDBJAHSDB,AHDB {tag_map}")
        print(f"THIS IS THE DATA DATA DTAA {data.columns}")
        if data.empty or (len(data) == 1 and data.iloc[0].get("Agent") == "No data"):
            logger.warning("No data available to view. Returning empty DataFrame and plots.")
            return data, {}

        base_cols = ["Agent", "Submitter", "Date", "Logs"]
        existing_cols = [col for col in base_cols if col in data.columns]

        primary_score_col: str
        group_metric_names: list[str]

        if tag is None:
            primary = "Overall"
            group = list(tag_map.keys())
        else:
            primary = tag
            group = tag_map.get(tag, [])

        if f"{primary} Score" in data.columns:
            data = data.sort_values(f"{primary} Score", ascending=False)
        else:
            logger.warning(f"Primary metric '{primary}' for sorting not found. Data will not be sorted by it.")

        metrics_to_display = []
        if f"{primary} Cost" in data.columns:
            metrics_to_display.append(f"{primary} Cost")
        if f"{primary} Score" in data.columns:
            metrics_to_display.append(f"{primary} Score")

        for g_item in group:
            if g_item in data.columns:
                metrics_to_display.append(g_item)
            if f"{g_item} Cost" in data.columns:
                metrics_to_display.append(f"{g_item} Cost")
            if f"{g_item} Score" in data.columns:
                metrics_to_display.append(f"{g_item} Score")


        final_cols_to_display = existing_cols + [m for m in metrics_to_display if m in data.columns]
        final_cols_to_display = sorted(list(set(final_cols_to_display)), key=final_cols_to_display.index)

        df_view = data.loc[:, final_cols_to_display].reset_index(drop=True)

        plots: dict[str, Any] = {}
        if with_plots:
            plot_metric_names = [primary] + [g_item for g_item in group if g_item in data.columns]
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


def _safe_round(value, digits=2):
    return round(value, digits) if isinstance(value, (float, int)) and pd.notna(value) else value

def _get_dataframe(
        records_list: list[dict],
        split: str,
        is_internal: bool,
        suite_config: SuiteConfig,
        timezone: str = "US/Pacific",
) -> pd.DataFrame:
    # This function remains the same as in the previous version you provided.
    # It takes a list of records (which will be a list containing one item
    # from the loaded JSON file) and processes it.
    if not records_list:
        logger.warning(f"No records provided to _get_dataframe for split '{split}'. Returning empty DataFrame with placeholder.")
        expected_pretty_cols = ["Agent Name", "Submitter", "Date", "Overall Score", "Logs"]
        empty_df = pd.DataFrame({p_col: ["No data"] for p_col in expected_pretty_cols})
        return empty_df

    cfg = suite_config

    rows = []
    for itm_idx, itm in enumerate(records_list):
        if not isinstance(itm, dict):
            logger.warning(f"Item {itm_idx} in records_list is not a dict, skipping.")
            continue
        try:
            ev = EvalResult.model_validate(itm)
        except Exception as e:
            logger.error(f"Failed to validate item {itm_idx} with EvalResult: {itm}. Error: {e}")
            continue

        sub = ev.submission
        date_str = None
        if sub.submit_time is not None:
            submit_dt = sub.submit_time
            if not isinstance(submit_dt, pd.Timestamp):
                if submit_dt.tzinfo is None:
                    logger.debug(f"Submission time for {sub.agent_name} is timezone-naive, assuming UTC.")
                    submit_dt = submit_dt.replace(tzinfo=ZoneInfo("UTC"))
            date_str = pd.Timestamp(submit_dt).tz_convert(ZoneInfo(timezone)).strftime("%Y-%m-%d")
        else:
            date_str = None

        if not ev.results:
            logger.warning(
                f"Skipping submission {sub.agent_name} ({sub.username or 'N/A'}) "
                f"({sub.submit_time or 'N/A'}) due to no results."
            )
            continue
        stats = compute_summary_statistics(
            suite_config=cfg, split=split, results=ev.results
        )
        flat = {}
        print(f"STATS STATS ASTATAS SD T S T A A {stats}")
        for key, s_obj in stats.items():
            parts = key.split("/")
            if parts[0] == "overall":
                flat["overall/score"] = _safe_round(getattr(s_obj, 'score', np.nan))
                flat["overall/cost"] = _safe_round(getattr(s_obj, 'cost', np.nan))
            elif parts[0] == "tag" and len(parts) > 1:
                tag_name = parts[1]
                flat[f"tag/{tag_name}/score"] = _safe_round(getattr(s_obj, 'score', np.nan))
                flat[f"tag/{tag_name}/cost"] = _safe_round(getattr(s_obj, 'cost', np.nan))
            elif parts[0] == "task" and len(parts) > 1:
                task_name = parts[1]
                score = getattr(s_obj, 'score', np.nan)
                cost = getattr(s_obj, 'cost', np.nan)
                score_stderr = getattr(s_obj, 'score_stderr', np.nan)
                cost_stderr = getattr(s_obj, 'cost_stderr', np.nan)

                flat[f"task/{task_name}/score"] = _safe_round(score)
                flat[f"task/{task_name}/score_ci"] = _safe_round(score_stderr * 1.96 if pd.notna(score_stderr) else np.nan)
                flat[f"task/{task_name}/cost"] = _safe_round(cost)
                flat[f"task/{task_name}/cost_ci"] = _safe_round(cost_stderr * 1.96 if pd.notna(cost_stderr) else np.nan)
            else:
                logger.debug(f"Uncommon key structure from compute_summary_statistics: '{key}'. Attempting generic add.")
                if hasattr(s_obj, 'score'):
                    flat[f"{key}/score"] = _safe_round(s_obj.score)
                if hasattr(s_obj, 'cost'):
                    flat[f"{key}/cost"] = _safe_round(s_obj.cost)

        current_logs_url = None
        if is_internal and sub.logs_url:
            current_logs_url = str(sub.logs_url)
        elif not is_internal and sub.logs_url_public:
            current_logs_url = str(sub.logs_url_public)

        rows.append(
            {
                "agent_name": sub.agent_name or "N/A",
                "username": sub.username or "N/A",
                "submit_time": date_str,
                **flat,
                "logs_url": current_logs_url,
            }
        )

    if not rows:
        logger.warning(f"No valid rows generated from records_list for split '{split}'. Returning empty DataFrame with placeholder.")
        expected_pretty_cols = ["Agent", "Submitter", "Date", "Overall Score", "Overall Cost", "Logs"]
        empty_df = pd.DataFrame({p_col: ["No data"] for p_col in expected_pretty_cols})
        return empty_df

    df = pd.DataFrame(rows)
    pretty_cols = {c: _pretty_column_name(c) for c in df.columns if c in df.columns}
    overview = df.rename(columns=pretty_cols)
    return overview

def _pretty_column_name(col: str) -> str:
    """Map raw column name to display name."""
    # --- Step 1: Fixed, direct mappings ---
    fixed_mappings = {
        "submit_time": "Date",
        "agent_name": "Agent",
        "username": "Submitter",
        "logs_url": "Logs",
        "overall/score": "Overall Score",
        "overall/cost": "Overall Cost",
    }
    if col in fixed_mappings:
        return fixed_mappings[col]

    # --- Step 2: Define your mapping for informal names to descriptive names ---
    informal_map = DataTransformer._INFORMAL_TO_FORMAL_NAME_MAP

    # --- Step 3: Dynamic mappings for task or tag columns using the informal_to_formal_name_map ---
    parts = col.split("/")
    if len(parts) == 3:
        item_type, informal_name, metric_suffix = parts #

        formal_name = informal_map.get(informal_name)
        if formal_name is None:
            formal_name = informal_name.replace("_", " ").title()
            print(f"[DEBUG _pretty_column_name] Informal name '{informal_name}' not in map, using fallback: '{formal_name}'")

        if metric_suffix == "score":
            return f"{formal_name} Score"
        if metric_suffix == "cost":
            return f"{formal_name} Cost"
        if metric_suffix == "score_ci":
            return f"{formal_name} Score 95% CI"
        if metric_suffix == "cost_ci":
            return f"{formal_name} Cost 95% CI"

    # --- Step 4: Fallback for columns that don't match the "type/name/metric" pattern ---
    if "/" not in col:
        return col.replace("_", " ").title()
    else:
        return parts[-1].replace("_", " ").title()

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