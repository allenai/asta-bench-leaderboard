import gradio as gr
from gradio.events import SelectData
import pandas as pd
import plotly.graph_objects as go
import os

from agenteval.leaderboard.view import LeaderboardViewer
from huggingface_hub import HfApi

from leaderboard_transformer import (
    DataTransformer,
    transform_raw_dataframe,
    create_pretty_tag_map,
    INFORMAL_TO_FORMAL_NAME_MAP,
    _plot_scatter_plotly,
    format_cost_column,
    format_score_column,
    get_pareto_df,
)
from content import (
    SCATTER_DISCLAIMER,
    PARETO_DISCLAIMER,
    format_error,
    format_log,
    format_warning,
    hf_uri_to_web_url,
    hyperlink,
)

# --- Constants and Configuration  ---
LOCAL_DEBUG = not (os.environ.get("system") == "spaces")
CONFIG_NAME = "1.0.0-dev1" # This corresponds to 'config' in LeaderboardViewer
IS_INTERNAL = os.environ.get("IS_INTERNAL", "false").lower() == "true"

OWNER = "allenai"
PROJECT_NAME = "asta-bench" + ("-internal" if IS_INTERNAL else "")
SUBMISSION_DATASET = f"{OWNER}/{PROJECT_NAME}-submissions"
SUBMISSION_DATASET_PUBLIC = f"{OWNER}/{PROJECT_NAME}-submissions-public"
CONTACT_DATASET = f"{OWNER}/{PROJECT_NAME}-contact-info"
RESULTS_DATASET = f"{OWNER}/{PROJECT_NAME}-results" # This is the repo_id for LeaderboardViewer
LEADERBOARD_PATH = f"{OWNER}/{PROJECT_NAME}-leaderboard"

if LOCAL_DEBUG:
    DATA_DIR = os.path.join(os.path.dirname(__file__), "data", CONFIG_NAME)
else:
    DATA_DIR = "/home/user/data/" + CONFIG_NAME
EXTRACTED_DATA_DIR = os.path.join(DATA_DIR, "extracted")

api = HfApi()
MAX_UPLOAD_BYTES = 100 * 1024**2
AGENTEVAL_MANIFEST_NAME = "agenteval.json"
os.makedirs(EXTRACTED_DATA_DIR, exist_ok=True)


# --- Global State for Viewers (simple caching) ---
CACHED_VIEWERS = {}
CACHED_TAG_MAPS = {}

# --- New Helper Class to Solve the Type Mismatch Bug ---
class DummyViewer:
    """A mock viewer to be cached on error. It has a ._load() method
       to ensure it behaves like the real LeaderboardViewer."""
    def __init__(self, error_df):
        self._error_df = error_df

    def _load(self):
        # The _load method returns the error DataFrame and an empty tag map
        return self._error_df, {}

def get_leaderboard_viewer_instance(split: str):
    """
    Fetches the LeaderboardViewer for a split, using a cache to avoid
    re-downloading data. On error, returns a stable DummyViewer object.
    """
    global CACHED_VIEWERS, CACHED_TAG_MAPS

    if split in CACHED_VIEWERS:
        # Cache hit: return the cached viewer and tag map
        return CACHED_VIEWERS[split], CACHED_TAG_MAPS.get(split, {"Overall": []})

    # --- Cache miss: try to load data from the source ---
    try:
        print(f"Using Hugging Face dataset for split '{split}': {RESULTS_DATASET}/{CONFIG_NAME}")
        viewer = LeaderboardViewer(
            repo_id=RESULTS_DATASET,
            config=CONFIG_NAME,
            split=split,
            is_internal=IS_INTERNAL
        )

        # Simplify tag map creation
        pretty_tag_map = create_pretty_tag_map(viewer.tag_map, INFORMAL_TO_FORMAL_NAME_MAP)

        # Cache the results for next time
        CACHED_VIEWERS[split] = viewer
        CACHED_TAG_MAPS[split] = pretty_tag_map # Cache the pretty map directly

        return viewer, pretty_tag_map

    except Exception as e:
        # On ANY error, create a consistent error message and cache a DummyViewer
        error_message = f"Error loading data for split '{split}': {e}"
        print(format_error(error_message))

        dummy_df = pd.DataFrame({"Message": [error_message]})
        dummy_viewer = DummyViewer(dummy_df)
        dummy_tag_map = {"Overall": []}

        # Cache the dummy objects so we don't try to fetch again on this run
        CACHED_VIEWERS[split] = dummy_viewer
        CACHED_TAG_MAPS[split] = dummy_tag_map

        return dummy_viewer, dummy_tag_map


def create_leaderboard_display(
        full_df: pd.DataFrame,
        tag_map: dict,
        category_name: str,
        split_name: str
):
    """
    This UI factory takes pre-loaded data and renders the main DataFrame and Plot
    for a given category (e.g., "Overall" or "Literature Understanding").
    """
    # 1. Instantiate the transformer and get the specific view for this category.
    # The function no longer loads data itself; it filters the data it receives.
    transformer = DataTransformer(full_df, tag_map)
    df_view, plots_dict = transformer.view(tag=category_name, use_plotly=True)
    pareto_df = get_pareto_df(df_view)
    # Get the list of agents on the frontier. We'll use this list later.
    if not pareto_df.empty and 'Agent' in pareto_df.columns:
        pareto_agent_names = pareto_df['Agent'].tolist()
    else:
        pareto_agent_names = []

    # Format cost columns
    for col in df_view.columns:
        if "Cost" in col:
            df_view = format_cost_column(df_view, col)

    # Fill NaN scores with 0
    for col in df_view.columns:
        if "Score" in col:
            df_view = format_score_column(df_view, col)
    scatter_plot = plots_dict.get('scatter_plot', go.Figure())

    # Highlight agents on the Pareto frontier in the DataFrame.
    styled_data = []
    df_headers = df_view.columns.tolist()
    try:
        agent_col_index = df_headers.index('Agent')
    except ValueError:
        agent_col_index = -1 # Agent column not found

    for _, row in df_view.iterrows():
        new_row = []
        for i, item in enumerate(row):
            # Check if this is the 'Agent' column AND if this agent is on the frontier.
            if i == agent_col_index and item in pareto_agent_names:
                styled_item = f"<span style='color: #22c55e; font-weight: bold;'>{item}</span>"
                new_row.append(styled_item)
            else:
                new_row.append(str(item) if pd.notna(item) else "")
        styled_data.append(new_row)
    df_datatypes = ["markdown" if col == "Logs" or col == "Agent" or "Cost" in col or "Score" in col else "str" for col in df_headers]

    plot_component = gr.Plot(
        value=scatter_plot,
        label=f"Score vs. Cost ({category_name})"
    )
    gr.HTML(SCATTER_DISCLAIMER, elem_id="scatter-disclaimer")
    dataframe_component = gr.DataFrame(
        headers=df_headers,
        value=styled_data,
        datatype=df_datatypes,
        interactive=False,
        wrap=True,
        column_widths=[100, 100, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 75, 75, 50, 50]
    )
    gr.HTML(PARETO_DISCLAIMER, elem_id="pareto-disclaimer")

    # Return the components so they can be referenced elsewhere.
    return plot_component, dataframe_component,

def get_full_leaderboard_data(split: str) -> tuple[pd.DataFrame, dict]:
    """
    Loads and transforms the complete dataset for a given split.
    This function handles caching and returns the final "pretty" DataFrame and tag map.
    """
    # This reuses your existing robust caching logic
    viewer_or_data, raw_tag_map = get_leaderboard_viewer_instance(split)

    if isinstance(viewer_or_data, (LeaderboardViewer, DummyViewer)):
        raw_df, _ = viewer_or_data._load()
        if raw_df.empty:
            return pd.DataFrame(), {}

        pretty_df = transform_raw_dataframe(raw_df)
        pretty_tag_map = create_pretty_tag_map(raw_tag_map, INFORMAL_TO_FORMAL_NAME_MAP)
        if "Logs" in pretty_df.columns:
            def format_log_entry_to_html(raw_uri):
                if pd.isna(raw_uri) or raw_uri == "": return ""
                web_url = hf_uri_to_web_url(str(raw_uri))
                return hyperlink(web_url, "🔗") if web_url else ""

            # Apply the function to the "Logs" column
            pretty_df["Logs"] = pretty_df["Logs"].apply(format_log_entry_to_html)

        return pretty_df, pretty_tag_map

    # Fallback for unexpected types
    return pd.DataFrame(), {}

# --- Detailed Benchmark Display ---
def create_benchmark_details_display(
        full_df: pd.DataFrame,
        tag_map: dict,
        category_name: str
):
    """
    Generates a detailed breakdown for each benchmark within a given category.
    For each benchmark, it creates a title, a filtered table, and a scatter plot.
    Args:
        full_df (pd.DataFrame): The complete, "pretty" dataframe for the entire split.
        tag_map (dict): The "pretty" tag map to find the list of benchmarks.
        category_name (str): The main category to display details for (e.g., "Literature Understanding").
    """
    # 1. Get the list of benchmarks for the selected category
    benchmark_names = tag_map.get(category_name, [])

    if not benchmark_names:
        gr.Markdown(f"No detailed benchmarks found for the category: {category_name}")
        return

    gr.Markdown("---")
    gr.Markdown("## Detailed Benchmark Results")

    # 2. Loop through each benchmark and create its UI components
    for benchmark_name in benchmark_names:
        with gr.Blocks():
            gr.Markdown(f"### {benchmark_name}")

            # 3. Prepare the data for this specific benchmark's table and plot
            benchmark_score_col = f"{benchmark_name} Score"
            benchmark_cost_col = f"{benchmark_name} Cost"

            # Define the columns needed for the detailed table
            table_cols = ['Agent', 'Submitter', 'Date', benchmark_score_col, benchmark_cost_col,'Logs']

            # Filter to only columns that actually exist in the full dataframe
            existing_table_cols = [col for col in table_cols if col in full_df.columns]

            if benchmark_score_col not in existing_table_cols:
                gr.Markdown(f"Score data for {benchmark_name} not available.")
                continue # Skip to the next benchmark if score is missing

            # Create a specific DataFrame for the table view
            benchmark_table_df = full_df[existing_table_cols].copy()
            pareto_df = get_pareto_df(benchmark_table_df)
            # Get the list of agents on the frontier. We'll use this list later.
            if not pareto_df.empty and 'Agent' in pareto_df.columns:
                pareto_agent_names = pareto_df['Agent'].tolist()
            else:
                pareto_agent_names = []
            # Calculated and add "Benchmark Attempted" column
            def check_benchmark_status(row):
                has_score = pd.notna(row.get(benchmark_score_col))
                has_cost = pd.notna(row.get(benchmark_cost_col))

                if has_score and has_cost:
                    return "✅"
                if has_score or has_cost:
                    return "⚠️"
                return "🚫 "

            # Apply the function to create the new column
            benchmark_table_df['Attempted Benchmark'] = benchmark_table_df.apply(check_benchmark_status, axis=1)
            # Sort the DataFrame
            if benchmark_score_col in benchmark_table_df.columns:
                benchmark_table_df = benchmark_table_df.sort_values(
                    by=benchmark_score_col, ascending=False, na_position='last'
                )
            # 1. Format the cost and score columns
            benchmark_table_df = format_cost_column(benchmark_table_df, benchmark_cost_col)
            benchmark_table_df = format_score_column(benchmark_table_df, benchmark_score_col)
            desired_cols_in_order = [
                'Agent',
                'Submitter',
                'Attempted Benchmark',
                benchmark_score_col,
                benchmark_cost_col,
                'Openness',
                'Degree of Control',
                'Date',
                'Logs'
            ]
            for col in desired_cols_in_order:
                if col not in benchmark_table_df.columns:
                    benchmark_table_df[col] = pd.NA # Add as an empty column
            benchmark_table_df = benchmark_table_df[desired_cols_in_order]
            # Rename columns for a cleaner table display, as requested
            benchmark_table_df.rename(columns={
                benchmark_score_col: 'Score',
                benchmark_cost_col: 'Cost'
            }, inplace=True)
            # Ensure the 'Logs' column is formatted correctly

            # Highlight agents on the Pareto frontier in the DataFrame.
            styled_data = []
            df_headers = benchmark_table_df.columns.tolist()
            try:
                agent_col_index = df_headers.index('Agent')
            except ValueError:
                agent_col_index = -1 # Agent column not found

            for _, row in benchmark_table_df.iterrows():
                new_row = []
                for i, item in enumerate(row):
                    # Check if this is the 'Agent' column AND if this agent is on the frontier.
                    if i == agent_col_index and item in pareto_agent_names:
                        styled_item = f"<span style='color: #22c55e; font-weight: bold;'>{item}</span>"
                        new_row.append(styled_item)
                    else:
                        new_row.append(str(item) if pd.notna(item) else "")
                styled_data.append(new_row)
            df_datatypes = ["markdown" if col == "Logs" or col == "Agent" or "Cost" in col or "Score" in col else "str" for col in df_headers]

            # Create the scatter plot using the full data for context, but plotting benchmark metrics
            # This shows all agents on the same axis for better comparison.
            benchmark_plot = _plot_scatter_plotly(
                data=full_df,
                x=benchmark_cost_col,
                y=benchmark_score_col,
                agent_col="Agent"
            )
            gr.Plot(value=benchmark_plot)
            gr.HTML(SCATTER_DISCLAIMER, elem_id="scatter-disclaimer")
            # Create the Gradio component, now with the correct datatypes
            gr.DataFrame(
                headers=df_headers,
                value=styled_data,
                datatype=df_datatypes,
                interactive=False,
                wrap=True,
            )
            gr.HTML(PARETO_DISCLAIMER, elem_id="pareto-disclaimer")
