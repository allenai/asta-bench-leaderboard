import gradio as gr
import pandas as pd
import plotly.graph_objects as go
import os
import re
import base64

from agenteval.leaderboard.view import LeaderboardViewer
from huggingface_hub import HfApi

import aliases
from leaderboard_transformer import (
    DataTransformer,
    transform_raw_dataframe,
    create_pretty_tag_map,
    INFORMAL_TO_FORMAL_NAME_MAP,
    _plot_scatter_plotly,
    format_cost_column,
    format_score_column,
    get_pareto_df,
    clean_llm_base_list,
)
from config import (
    CONFIG_NAME,
    EXTRACTED_DATA_DIR,
    IS_INTERNAL,
    RESULTS_DATASET,
)
from content import (
    scatter_disclaimer_html,
    format_error,
    format_log,
    format_warning,
    hf_uri_to_web_url,
    hyperlink,
)

api = HfApi()
MAX_UPLOAD_BYTES = 100 * 1024**2
AGENTEVAL_MANIFEST_NAME = "agenteval.json"
os.makedirs(EXTRACTED_DATA_DIR, exist_ok=True)
# Global variables
COMBINED_ICON_MAP = {
    "Open Source + Open Weights": {
        "Standard": "assets/os-ow-standard.svg",        # Bright pink star
        "Custom with Standard Search": "assets/os-ow-equivalent.svg",    # Bright pink diamond
        "Fully Custom": "assets/os-ow-custom.svg",            # Bright pink triangle
    },
    "Open Source": {
        "Standard": "assets/os-standard.svg",        # Orange/pink star
        "Custom with Standard Search": "assets/os-equivalent.svg",    # Orange/pink diamond
        "Fully Custom": "assets/os-custom.svg",            # Orange/pink triangle
    },
    "API Available": {
        "Standard": "assets/api-standard.svg",       # Yellow/pink star
        "Custom with Standard Search": "assets/api-equivalent.svg",   # Yellow/pink diamond
        "Fully Custom": "assets/api-custom.svg",           # Yellow/pink triangle
    },
    "Closed": {
        "Standard": "assets/c-standard.svg",        # Hollow pink star
        "Custom with Standard Search": "assets/c-equivalent.svg",    # Hollow pink diamond
        "Fully Custom": "assets/c-custom.svg",            # Hollow pink triangle
    }
}


# it's important to do the tool usage first here, so that when
# we do openness, the tool usage changes get picked up
for openness in COMBINED_ICON_MAP:
    for canonical_tool_usage, tool_usage_aliases in aliases.TOOL_USAGE_ALIASES.items():
        for tool_usage_alias in tool_usage_aliases:
            COMBINED_ICON_MAP[openness][tool_usage_alias] = COMBINED_ICON_MAP[openness][canonical_tool_usage]

for canonical_openness, openness_aliases in aliases.OPENNESS_ALIASES.items():
    for openness_alias in openness_aliases:
        COMBINED_ICON_MAP[openness_alias] = COMBINED_ICON_MAP[canonical_openness]


OPENNESS_SVG_MAP = {
    "Open Source + Open Weights": "assets/os-ow-standard.svg",
    "Open Source": "assets/os-standard.svg",
    "API Available": "assets/api-standard.svg",
    "Closed": "assets/c-standard.svg",
}
TOOLING_SVG_MAP = {
    "Standard": "assets/os-ow-standard.svg",
    "Custom with Standard Search": "assets/os-ow-equivalent.svg",
    "Fully Custom": "assets/os-ow-custom.svg",
}

def get_svg_as_data_uri(path: str) -> str:
    """Reads an SVG file and returns it as a base64-encoded data URI."""
    try:
        with open(path, "rb") as svg_file:
            encoded_svg = base64.b64encode(svg_file.read()).decode("utf-8")
            return f"data:image/svg+xml;base64,{encoded_svg}"
    except FileNotFoundError:
        print(f"Warning: SVG file not found at {path}")
        return ""

# Create a pre-loaded version of our map. This should be run ONCE when the app starts.
PRELOADED_URI_MAP = {
    openness: {
        tooling: get_svg_as_data_uri(path)
        for tooling, path in tooling_map.items()
    }
    for openness, tooling_map in COMBINED_ICON_MAP.items()
}

def get_combined_icon_html(row, uri_map):
    """
    Looks up the correct icon URI from the pre-loaded map based on the row's
    'Openness' and 'Agent Tooling' values and returns an HTML <img> tag.
    """
    openness_val = row['Openness']
    tooling_val = row['Agent Tooling']
    uri = uri_map.get(openness_val, {}).get(tooling_val, "")
    # The tooltip will show the exact combination for clarity.
    tooltip = f"Openness: {openness_val}, Tooling: {tooling_val}"

    # Return the HTML string that Gradio will render in the DataFrame.
    return f'<img src="{uri}" alt="{tooltip}" title="{tooltip}" style="width:24px; height:24px;">'

def create_svg_html(value, svg_map):
    """
    Generates the absolute simplest HTML for an icon, without any extra text.
    This version is compatible with gr.DataFrame.
    """
    if pd.isna(value) or value not in svg_map:
        return ""

    path_info = svg_map[value]

    src = get_svg_as_data_uri(path_info)
    # Generate the HTML for the single icon, with NO text.
    if src:
        return f'<img src="{src}" style="width: 16px; height: 16px; vertical-align: middle;" alt="{value}" title="{value}">'
    return ""

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
for name, path in TOOLING_SVG_MAP.items():
    uri = get_svg_as_data_uri(path)
    tooling_html_items.append(
        f'<div style="display: flex; align-items: center; white-space: nowrap;">'
        f'<img src="{uri}" alt="{name}" title="{name}" style="width:16px; height:16px; margin-right: 4px; flex-shrink: 0;">'
        f'<span>{name}</span>'
        f'</div>'
    )
tooling_html = " ".join(tooling_html_items)


# Your final legend_markdown string (the structure of this does not change)
legend_markdown = f"""
<div style="display: flex; flex-wrap: wrap; align-items: flex-start; gap: 24px; font-size: 14px; padding-bottom: 8px;">
        
    <div> <!-- Container for the Pareto section -->
        <b>Pareto</b><span class="tooltip-icon" data-tooltip="Indicates if agent is on the Pareto frontier
        ">ⓘ</span>
        <div style="padding-top: 4px;"><span>🏆 On frontier</span></div>
    </div>

    <div> <!-- Container for the Openness section -->
        <b>Agent Openness</b><span class="tooltip-icon" data-tooltip="•Closed: No API or code available
        •API Available: API available, but no code
        •Open Source: Code available, but no weights
        •Open Source + Open Weights: Code and weights available
        ">ⓘ</span>
        <div style="display: flex; flex-wrap: wrap; align-items: center; gap: 16px; margin-top: 4px;">{openness_html}</div>
    </div>

    <div> <!-- Container for the Tooling section -->
        <b>Agent Tooling</b><span class="tooltip-icon" data-tooltip="• Standard: Standard Approach used by the agent
        • Custom with Standard Search: Standard search used by the agent
        • Fully Custom: Fully custom tools used by the agent
        ">ⓘ</span>
        <div style="display: flex; flex-wrap: wrap; align-items: center; gap: 16px; margin-top: 4px;">{tooling_html}</div>
    </div>
    
     <div><b>Column Descriptions</b><span class="tooltip-icon" data-tooltip="• Overall Score: Performance across all benchmarks
        • Overall Cost: Cost per task in USD
        • Literature Understanding Score: Performance on scientific literature tasks
        • Literature Understanding Cost: Cost per literature understanding task in USD
        • Data Analysis Score: Performance on data analysis tasks
        • Code Execution Score: Performance on coding tasks
        • Code Execution Cost: Cost per code execution task in USD
        • Discovery Score: Performance on information discovery tasks
        • Discovery Cost: Cost per discovery task in USD
        • Categories Attempted: Number of benchmark categories the agent participated in
        • Logs: Link to detailed evaluation logs">ⓘ</span></div>
</div>
"""

# --- Global State for Viewers (simple caching) ---
CACHED_VIEWERS = {}
CACHED_TAG_MAPS = {}


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
        lambda row: '🏆' if row['id'] in pareto_agent_names else '',
        axis=1
    )
    # Create mapping for Openness / tooling
    df_view['Icon'] = df_view.apply(
        lambda row: get_combined_icon_html(row, PRELOADED_URI_MAP),
        axis=1  # IMPORTANT: axis=1 tells pandas to process row-by-row
    )

    # Format cost columns
    for col in df_view.columns:
        if "Cost" in col:
            df_view = format_cost_column(df_view, col)

    # Fill NaN scores with 0
    for col in df_view.columns:
        if "Score" in col:
            df_view = format_score_column(df_view, col)
    scatter_plot = plots_dict.get('scatter_plot', go.Figure())
    #Make pretty and format the LLM Base column
    df_view['LLM Base'] = df_view['LLM Base'].apply(clean_llm_base_list)
    df_view['LLM Base'] = df_view['LLM Base'].apply(format_llm_base_with_html)

    all_cols = df_view.columns.tolist()
    # Remove pareto and Icon columns and insert it at the beginning
    all_cols.insert(0, all_cols.pop(all_cols.index('Icon')))
    all_cols.insert(0, all_cols.pop(all_cols.index('Pareto')))
    df_view = df_view[all_cols]
    # Drop internally used columns that are not needed in the display
    columns_to_drop = ['id', 'agent_for_hover', 'Openness', 'Agent Tooling']
    df_view = df_view.drop(columns=columns_to_drop, errors='ignore')

    df_headers = df_view.columns.tolist()
    df_datatypes = []
    for col in df_headers:
        if col in ["Logs", "Agent"] or "Cost" in col or "Score" in col:
            df_datatypes.append("markdown")
        elif col in ["Icon","LLM Base"]:
            df_datatypes.append("html")
        else:
            df_datatypes.append("str")

    header_rename_map = {
        "Pareto": "",
        "Icon": "",
    }
    # 2. Create the final list of headers for display.
    df_view = df_view.rename(columns=header_rename_map)

    plot_component = gr.Plot(
        value=scatter_plot,
        show_label=False
    )
    gr.HTML(value=scatter_disclaimer_html, elem_id="scatter-disclaimer")
    # Put table and key into an accordion
    with gr.Accordion("Show / Hide Table View", open=True, elem_id="leaderboard-accordion"):
        gr.HTML(value=legend_markdown, elem_id="legend-markdown")
        dataframe_component = gr.DataFrame(
            headers=df_headers,
            value=df_view,
            datatype=df_datatypes,
            interactive=False,
            wrap=True,
            column_widths=[40, 40, 200, 200],
            elem_classes=["wrap-header-df"]
        )

    # Return the components so they can be referenced elsewhere.
    return plot_component, dataframe_component

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
        with gr.Row(elem_classes=["benchmark-header"]):
            gr.Markdown(f"### {benchmark_name} Leaderboard", header_links=True)
            button_str = f"""
            <button
                class="scroll-up-button"
                onclick="scroll_to_element('page-content-wrapper')"
            >
                {"⬆"}
            </button>
            """
            gr.HTML(button_str,elem_classes="scroll-up-container")

        # 3. Prepare the data for this specific benchmark's table and plot
        benchmark_score_col = f"{benchmark_name} Score"
        benchmark_cost_col = f"{benchmark_name} Cost"

        # Define the columns needed for the detailed table
        table_cols = ['Agent','Openness','Agent Tooling', 'Submitter', 'Date', benchmark_score_col, benchmark_cost_col,'Logs','id', 'LLM Base']

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
            lambda row: ' 🏆' if row['id'] in pareto_agent_names else '',
            axis=1
        )

        benchmark_table_df['Icon'] = benchmark_table_df.apply(
            lambda row: get_combined_icon_html(row, PRELOADED_URI_MAP),
            axis=1  # IMPORTANT: axis=1 tells pandas to process row-by-row
        )

        #Make pretty and format the LLM Base column
        benchmark_table_df['LLM Base'] = benchmark_table_df['LLM Base'].apply(clean_llm_base_list)
        benchmark_table_df['LLM Base'] = benchmark_table_df['LLM Base'].apply(format_llm_base_with_html)

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
            'Icon',
            'Agent',
            'Submitter',
            'LLM Base',
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
            elif col in ["Icon", "LLM Base"]:
                df_datatypes.append("html")
            else:
                df_datatypes.append("str")
        # Remove Pareto, Openness, and Agent Tooling from the headers
        header_rename_map = {
            "Pareto": "",
            "Icon": "",
        }
        # 2. Create the final list of headers for display.
        benchmark_table_df = benchmark_table_df.rename(columns=header_rename_map)
        # Create the scatter plot using the full data for context, but plotting benchmark metrics
        # This shows all agents on the same axis for better comparison.
        benchmark_plot = _plot_scatter_plotly(
            data=full_df,
            x=benchmark_cost_col,
            y=benchmark_score_col,
            agent_col="Agent",
            name=benchmark_name
        )
        gr.Plot(value=benchmark_plot, show_label=False)
        gr.HTML(value=scatter_disclaimer_html, elem_id="scatter-disclaimer")
        # Put table and key into an accordion
        with gr.Accordion("Show / Hide Table View", open=True, elem_id="leaderboard-accordion"):
            gr.HTML(value=legend_markdown, elem_id="legend-markdown")
            gr.DataFrame(
                headers=df_headers,
                value=benchmark_table_df,
                datatype=df_datatypes,
                interactive=False,
                wrap=True,
                column_widths=[40, 40, 200, 150, 175, 85],
                elem_classes=["wrap-header-df"]
            )

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
def create_gradio_anchor_id(text: str, validation) -> str:
    """
    Replicates the ID format created by gr.Markdown(header_links=True).
    Example: "Paper Finder Validation" -> "h-paper-finder-validation"
    """
    text = text.lower()
    text = re.sub(r'\s+', '-', text) # Replace spaces with hyphens
    text = re.sub(r'[^\w-]', '', text) # Remove non-word characters
    if validation:
        return f"h-{text}-leaderboard-1"
    return f"h-{text}-leaderboard"
def create_sub_navigation_bar(tag_map: dict, category_name: str, validation: bool = False) -> gr.HTML:
    """
    Builds the entire sub-navigation bar as a single, self-contained HTML component.
    This bypasses Gradio's layout components, giving us full control.
    """
    benchmark_names = tag_map.get(category_name, [])
    if not benchmark_names:
        # Return an empty HTML component to prevent errors
        return gr.HTML()

    # Start building the list of HTML button elements as strings
    html_buttons = []
    for name in benchmark_names:
        target_id = create_gradio_anchor_id(name, validation)

        # Create a standard HTML button.
        # The onclick attribute calls our global JS function directly.
        # Note the mix of double and single quotes.
        button_str = f"""
            <button
                class="sub-nav-link-button"
                onclick="scroll_to_element('{target_id}')"
            >
                {name}
            </button>
        """
        html_buttons.append(button_str)

    # Join the button strings and wrap them in a single div container
    # This container will be our flexbox row.
    full_html = f"""
        <div class="sub-nav-bar-container">
            <span class="sub-nav-label">Benchmarks:</span>
            {''.join(html_buttons)}
        </div>
    """

    # Return the entire navigation bar as one single Gradio HTML component
    return gr.HTML(full_html)

def format_llm_base_with_html(value):
    """
    Formats the 'LLM Base' cell value.
    If the value is a list with more than 1 element, it returns an
      HTML <span> with the full list in a hover-over tooltip.
    If it's a single-element list, it returns just that element.
    Otherwise, it returns the original value.
    """
    if isinstance(value, list):
        if len(value) > 1:
            # Join the list items with a newline character for a clean tooltip
            tooltip_text = "\n".join(map(str, value))
            # Return an HTML span with the title attribute for the tooltip
            return f'<span class="tooltip-icon cell-tooltip-icon" style="cursor: help;" data-tooltip="{tooltip_text}">{value[0]} (+ {len(value) - 1}) ⓘ</span>'
        if len(value) == 1:
            # If only one item, just return that item
            return value[0]
    # Return the value as-is if it's not a list or is an empty list
    return value
