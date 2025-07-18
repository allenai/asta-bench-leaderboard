import gradio as gr
import pandas as pd
import plotly.graph_objects as go
import os
import re
import base64

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

# --- NEW: A global cache to store encoded SVG data ---
SVG_DATA_URI_CACHE = {}

def get_svg_as_data_uri(file_path: str) -> str:
    """
    Reads an SVG file, encodes it in Base64, and returns a Data URI.
    Uses a cache to avoid re-reading files from disk.
    """
    # Return from cache if we have already processed this file
    if file_path in SVG_DATA_URI_CACHE:
        return SVG_DATA_URI_CACHE[file_path]

    try:
        # Read the file in binary mode, encode it, and format as a Data URI
        with open(file_path, "rb") as svg_file:
            encoded_string = base64.b64encode(svg_file.read()).decode('utf-8')
        data_uri = f"data:image/svg+xml;base64,{encoded_string}"

        # Store in cache for future use
        SVG_DATA_URI_CACHE[file_path] = data_uri
        return data_uri
    except FileNotFoundError:
        # If the file doesn't exist, print a warning and return an empty string
        print(f"Warning: SVG file not found at '{file_path}'")
        return ""

def create_svg_html(value, svg_map):
    """
    Generates the absolute simplest HTML for an icon, without any extra text.
    This version is compatible with gr.DataFrame.
    """
    # If the value isn't in our map, return an empty string so the cell is blank.
    if pd.isna(value) or value not in svg_map:
        return ""

    path_info = svg_map[value]

    # For light/dark-aware icons (like Tooling)
    if isinstance(path_info, dict):
        light_theme_icon_uri = get_svg_as_data_uri(path_info['dark'])
        dark_theme_icon_uri = get_svg_as_data_uri(path_info['light'])

        # Generate the HTML for the two icons side-by-side, with NO text.
        img1 = f'<img src="{light_theme_icon_uri}" class="light-mode-icon" alt="{value}" title="{value}">'
        img2 = f'<img src="{dark_theme_icon_uri}" class="dark-mode-icon" alt="{value}" title="{value}">'
        return f'{img1}{img2}'

    # For single icons that don't change with theme (like Openness)
    elif isinstance(path_info, str):
        src = get_svg_as_data_uri(path_info)
        # Generate the HTML for the single icon, with NO text.
        return f'<img src="{src}" style="width: 16px; height: 16px; vertical-align: middle;" alt="{value}" title="{value}">'

    # Fallback in case of an unexpected data type
    return ""

# Global variables
OPENNESS_SVG_MAP = {
    "Closed": "assets/ui.svg", "API Available": "assets/api.svg", "Open Source": "assets/open-source.svg", "Open Source + Open Weights": "assets/open-weights.svg"
}
TOOLING_SVG_MAP = {
    "Standard": {"light": "assets/star-light.svg", "dark": "assets/star-dark.svg"},
    "Custom with Standard Search": {"light": "assets/diamond-light.svg", "dark": "assets/diamond-dark.svg"},
    "Fully Custom": {"light": "assets/circle-light.svg", "dark": "assets/circle-dark.svg"},
}

# Dynamically generate the correct HTML for the legend parts
openness_html = " ".join([create_svg_html(name, OPENNESS_SVG_MAP) for name in OPENNESS_SVG_MAP])
tooling_html = " ".join([create_svg_html(name, TOOLING_SVG_MAP) for name in TOOLING_SVG_MAP])
# Create HTML for the "Openness" legend items
openness_html_items = []
for name, path in OPENNESS_SVG_MAP.items():
    uri = get_svg_as_data_uri(path)
    # Each item is now its own flexbox container to guarantee alignment
    openness_html_items.append(
        f'<div style="display: flex; align-items: center; white-space: nowrap;">'
        f'<img src="{uri}" alt="{name}" title="{name}" style="width:16px; height:16px; margin-right: 4px; flex-shrink: 0;">'
        f'<span>{name}</span>'
        f'</div>'
    )
openness_html = " ".join(openness_html_items)

# Create HTML for the "Tooling" legend items
tooling_html_items = []
for name, paths in TOOLING_SVG_MAP.items():
    light_theme_icon_uri = get_svg_as_data_uri(paths['dark'])
    dark_theme_icon_uri = get_svg_as_data_uri(paths['light'])

    # The two swapping icons need to be stacked with absolute positioning
    img1 = f'<img src="{light_theme_icon_uri}" class="light-mode-icon" alt="{name}" title="{name}" style="position: absolute; top: 0; left: 0;">'
    img2 = f'<img src="{dark_theme_icon_uri}" class="dark-mode-icon" alt="{name}" title="{name}" style="position: absolute; top: 0; left: 0;">'

    # Their container needs a defined size and relative positioning
    icon_container = f'<div style="width: 16px; height: 16px; position: relative; flex-shrink: 0;">{img1}{img2}</div>'

    # This item is also a flexbox container
    tooling_html_items.append(
        f'<div style="display: flex; align-items: center; white-space: nowrap;">'
        f'{icon_container}'
        f'<span style="margin-left: 4px;">{name}</span>'
        f'</div>'
    )
tooling_html = " ".join(tooling_html_items)


# Your final legend_markdown string (the structure of this does not change)
legend_markdown = f"""
<div style="display: flex; flex-wrap: wrap; align-items: flex-start; gap: 24px; font-size: 14px; padding-bottom: 8px;">

    <div> <!-- Container for the Pareto section -->
        <b>Pareto</b>
        <div style="padding-top: 4px;"><span>📈 On frontier</span></div>
    </div>

    <div> <!-- Container for the Openness section -->
        <b>Agent Openness</b>
        <div style="display: flex; flex-wrap: wrap; align-items: center; gap: 16px; margin-top: 4px;">{openness_html}</div>
    </div>

    <div> <!-- Container for the Tooling section -->
        <b>Agent Tooling</b>
        <div style="display: flex; flex-wrap: wrap; align-items: center; gap: 16px; margin-top: 4px;">{tooling_html}</div>
    </div>

</div>
"""

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
    if not pareto_df.empty and 'id' in pareto_df.columns:
        pareto_agent_names = pareto_df['id'].tolist()
    else:
        pareto_agent_names = []
    df_view['Pareto'] = df_view.apply(
        lambda row: '📈' if row['id'] in pareto_agent_names else '',
        axis=1
    )
    # Create mapping for Openness / tooling
    df_view['Openness'] = df_view['Openness'].apply(lambda x: create_svg_html(x, OPENNESS_SVG_MAP))
    df_view['Agent Tooling'] = df_view['Agent Tooling'].apply(lambda x: create_svg_html(x, TOOLING_SVG_MAP))


    # Format cost columns
    for col in df_view.columns:
        if "Cost" in col:
            df_view = format_cost_column(df_view, col)

    # Fill NaN scores with 0
    for col in df_view.columns:
        if "Score" in col:
            df_view = format_score_column(df_view, col)
    scatter_plot = plots_dict.get('scatter_plot', go.Figure())


    all_cols = df_view.columns.tolist()
    # Remove 'Pareto' from the list and insert it at the beginning
    all_cols.insert(0, all_cols.pop(all_cols.index('Pareto')))
    df_view = df_view[all_cols]
    # Drop internally used columns that are not needed in the display
    columns_to_drop = ['id', 'agent_for_hover']
    df_view = df_view.drop(columns=columns_to_drop, errors='ignore')

    df_headers = df_view.columns.tolist()
    df_datatypes = []
    for col in df_headers:
        if col in ["Logs", "Agent"] or "Cost" in col or "Score" in col:
            df_datatypes.append("markdown")
        elif col in ["Openness", "Agent Tooling"]:
            df_datatypes.append("html")
        else:
            df_datatypes.append("str")

    header_rename_map = {
        "Pareto": "",
        "Openness": "",
        "Agent Tooling": ""
    }
    # 2. Create the final list of headers for display.
    df_view = df_view.rename(columns=header_rename_map)

    plot_component = gr.Plot(
        value=scatter_plot,
        label=f"Score vs. Cost ({category_name})"
    )
    gr.HTML(SCATTER_DISCLAIMER, elem_id="scatter-disclaimer")

    # Put table and key into an accordion
    with gr.Accordion("Details", open=True, elem_id="leaderboard-accordion"):
        gr.HTML(value=legend_markdown, elem_id="legend-markdown")
        dataframe_component = gr.DataFrame(
            headers=df_headers,
            value=df_view,
            datatype=df_datatypes,
            interactive=False,
            wrap=True,
            column_widths=[30, 30, 30, 250],
            elem_classes=["wrap-header-df"]
        )

    # Return the components so they can be referenced elsewhere.
    return plot_component, dataframe_component

def get_full_leaderboard_data(split: str) -> tuple[pd.DataFrame, dict]:
    """
    Loads and transforms the complete dataset for a given split.
    This function handles caching and returns the final "pretty" DataFrame and tag map.
    """
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
# Create sub-nav bar for benchmarks
def create_gradio_anchor_id(text: str) -> str:
    """
    Replicates the ID format created by gr.Markdown(header_links=True).
    Example: "Paper Finder Validation" -> "h-paper-finder-validation"
    """
    text = text.lower()
    text = re.sub(r'\s+', '-', text) # Replace spaces with hyphens
    text = re.sub(r'[^\w-]', '', text) # Remove non-word characters
    return f"h-{text}-leaderboard"
def create_sub_navigation_bar(tag_map: dict, category_name: str):
    """
    Generates and renders the HTML for the anchor-link sub-navigation bar.
    """
    benchmark_names = tag_map.get(category_name, [])
    if not benchmark_names:
        return # Do nothing if there are no benchmarks

    anchor_links = []
    for name in benchmark_names:
        # Use the helper function to create the correct ID format
        target_id = create_gradio_anchor_id(name)
        anchor_links.append(f"<a href='#{target_id}'>{name}</a>")

    nav_bar_html = f"<div class='sub-nav-bar'>{'   '.join(anchor_links)}</div>"

    # Use gr.HTML to render the links correctly
    gr.HTML(nav_bar_html)

# # --- Detailed Benchmark Display ---
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
            gr.Markdown(f"### {benchmark_name} Leaderboard", header_links=True)

            # 3. Prepare the data for this specific benchmark's table and plot
            benchmark_score_col = f"{benchmark_name} Score"
            benchmark_cost_col = f"{benchmark_name} Cost"

            # Define the columns needed for the detailed table
            table_cols = ['Agent','Openness','Agent Tooling', 'Submitter', 'Date', benchmark_score_col, benchmark_cost_col,'Logs','id']

            # Filter to only columns that actually exist in the full dataframe
            existing_table_cols = [col for col in table_cols if col in full_df.columns]

            if benchmark_score_col not in existing_table_cols:
                gr.Markdown(f"Score data for {benchmark_name} not available.")
                continue # Skip to the next benchmark if score is missing

            # Create a specific DataFrame for the table view
            benchmark_table_df = full_df[existing_table_cols].copy()
            pareto_df = get_pareto_df(benchmark_table_df)
            # Get the list of agents on the frontier. We'll use this list later.
            if not pareto_df.empty and 'id' in pareto_df.columns:
                pareto_agent_names = pareto_df['id'].tolist()
            else:
                pareto_agent_names = []
            benchmark_table_df['Pareto'] = benchmark_table_df.apply(
                lambda row: '📈' if row['id'] in pareto_agent_names else '',
                axis=1
            )

            benchmark_table_df['Openness'] = benchmark_table_df['Openness'].apply(lambda x: create_svg_html(x, OPENNESS_SVG_MAP))
            benchmark_table_df['Agent Tooling'] = benchmark_table_df['Agent Tooling'].apply(lambda x: create_svg_html(x, TOOLING_SVG_MAP))

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
                'Pareto',
                'Openness',
                'Agent Tooling',
                'Agent',
                'Submitter',
                'Attempted Benchmark',
                benchmark_score_col,
                benchmark_cost_col,
                'Logs'
            ]
            for col in desired_cols_in_order:
                if col not in benchmark_table_df.columns:
                    benchmark_table_df[col] = pd.NA # Add as an empty column
            benchmark_table_df = benchmark_table_df[desired_cols_in_order]
            # Rename columns for a cleaner table display, as requested
            benchmark_table_df.rename({
                benchmark_score_col: 'Score',
                benchmark_cost_col: 'Cost',
            }, inplace=True)
            # Ensure the 'Logs' column is formatted correctly
            df_headers = benchmark_table_df.columns.tolist()
            df_datatypes = []
            for col in df_headers:
                if "Logs" in col or "Cost" in col or "Score" in col:
                    df_datatypes.append("markdown")
                elif col in ["Openness", "Agent Tooling"]:
                    df_datatypes.append("html")
                else:
                    df_datatypes.append("str")

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
            # Put table and key into an accordion
            with gr.Accordion("Details", open=True, elem_id="leaderboard-accordion"):
                gr.HTML(value=legend_markdown, elem_id="legend-markdown")
                gr.DataFrame(
                    headers=df_headers,
                    value=benchmark_table_df,
                    datatype=df_datatypes,
                    interactive=False,
                    wrap=True,
                    elem_classes=["wrap-header-df"]
                )


