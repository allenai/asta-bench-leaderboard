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
    aliases.CANONICAL_OPENNESS_OPEN_OPEN_WEIGHTS: {
        aliases.CANONICAL_TOOL_USAGE_STANDARD: "assets/os-ow-standard.svg",
        aliases.CANONICAL_TOOL_USAGE_CUSTOM_INTERFACE: "assets/os-ow-equivalent.svg",
        aliases.CANONICAL_TOOL_USAGE_FULLY_CUSTOM: "assets/os-ow-custom.svg",
    },
    aliases.CANONICAL_OPENNESS_OPEN_CLOSED_WEIGHTS: {
        aliases.CANONICAL_TOOL_USAGE_STANDARD: "assets/os-standard.svg",
        aliases.CANONICAL_TOOL_USAGE_CUSTOM_INTERFACE: "assets/os-equivalent.svg",
        aliases.CANONICAL_TOOL_USAGE_FULLY_CUSTOM: "assets/os-custom.svg",
    },
    aliases.CANONICAL_OPENNESS_CLOSED_API_AVAILABLE: {
        aliases.CANONICAL_TOOL_USAGE_STANDARD: "assets/api-standard.svg",
        aliases.CANONICAL_TOOL_USAGE_CUSTOM_INTERFACE: "assets/api-equivalent.svg",
        aliases.CANONICAL_TOOL_USAGE_FULLY_CUSTOM: "assets/api-custom.svg",
    },
    aliases.CANONICAL_OPENNESS_CLOSED_UI_ONLY: {
        aliases.CANONICAL_TOOL_USAGE_STANDARD: "assets/c-standard.svg",
        aliases.CANONICAL_TOOL_USAGE_CUSTOM_INTERFACE: "assets/c-equivalent.svg",
        aliases.CANONICAL_TOOL_USAGE_FULLY_CUSTOM: "assets/c-custom.svg",
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
    "Open Source + Open Weights": "assets/os-ow-legend.svg",
    "Open Source": "assets/os-legend.svg",
    "API Available": "assets/api-legend.svg",
    "Closed": "assets/c-legend.svg",
}
TOOLING_SVG_MAP = {
    "Standard": "assets/standard-legend.svg",
    "Custom with Standard Search": "assets/equivalent-legend.svg",
    "Fully Custom": "assets/custom-legend.svg",
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

def build_openness_tooltip_content() -> str:
    """
    Generates the inner HTML for the Agent Openness tooltip card,
    """
    descriptions = {
        "Open Source + Open Weights": "Both code and ML models are open",
        "Open Source": "Code is open but uses an ML model with closed-weights",
        "API Available": "No access to code; API access only",
        "Closed": "No access to code or API; UI  access only",
    }
    html_items = []
    for name, path in OPENNESS_SVG_MAP.items():
        uri = get_svg_as_data_uri(path)
        desc = descriptions.get(name, "")

        # Create the HTML for a single row in the tooltip legend
        html_items.append(f"""
            <div class="tooltip-legend-item">
                <img src="{uri}" alt="{name}">
                <div>
                    <strong>{name}</strong>
                    <span>{desc}</span>
                </div>
            </div>
        """)

    return "".join(html_items)

def build_pareto_tooltip_content() -> str:
    """Generates the inner HTML for the Pareto tooltip card with final copy."""
    return f"""
        <h3>On Pareto Frontier</h3>
        <p class="tooltip-description">The Pareto frontier represents the best balance between score and cost.</p>
        <p class="tooltip-description">Agents on the frontier either:</p>
        <ul class="tooltip-sub-list">
            <li>Offer the lowest cost for a given performance, or</li>
            <li>Deliver the best performance at a given cost.</li>
        </ul>
        <p class="tooltip-description" style="margin-top: 12px;">These agents are marked with this icon: 🏆</p>
    """

def build_tooling_tooltip_content() -> str:
    """Generates the inner HTML for the Agent Tooling tooltip card."""
    descriptions = {
        "Standard": "Uses only predefined tools from the evaluation environment (as defined in Inspect's state.tools).",
        "Custom with Standard Search": "Custom tools for accessing an equivalent underlying environment:",
        "Fully Custom": "Uses tools beyond constraints of Standard or Custom interface",
    }
    custom_interface_sub_list = """
        <ul class="tooltip-sub-list">
            <li>Literature tasks: Information access is limited to date-restricted usage of the Asta MCP tools.</li>
            <li>Code tasks: Code execution is limited to an iPython shell in a machine environment initialized with the standard Asta sandbox Dockerfile (or equivalent).</li>
        </ul>
    """
    html_items = []
    for name, path in TOOLING_SVG_MAP.items():
        uri = get_svg_as_data_uri(path)
        desc = descriptions.get(name, "")

        # Check if this is the special case that needs a sub-list
        sub_list_html = custom_interface_sub_list if name == "Custom with Standard Search" else ""

        html_items.append(f"""
            <div class="tooltip-legend-item">
                <img src="{uri}" alt="{name}">
                <div>
                    <strong>{name}</strong>
                    <span>{desc}</span>
                    {sub_list_html}
                </div>
            </div>
        """)

    return "".join(html_items)


def build_descriptions_tooltip_content(table) -> str:
    """Generates the inner HTML for the Column Descriptions tooltip card depending on which kind of table."""
    if table == "Overall":
        return """
            <div class="tooltip-description-item"><b>Agent:</b> Name of the evaluated agent.</div>
            <div class="tooltip-description-item"><b>Submitter:</b> Organization or individual who submitted the agent for evaluation.</div>
            <div class="tooltip-description-item"><b>LLM Base:</b> Model(s) used by the agent. Hover over ⓘ to view all.</div>
            <div class="tooltip-description-item"><b>Overall Score:</b> Macro-average of the four category-level average scores. Each category contributes equally.</div>
            <div class="tooltip-description-item"><b>Overall Cost:</b> Macro-average cost per problem across all categories, in USD. Based on submission-time values. Each category contributes equally</div>
            <div class="tooltip-description-item"><b>Literature Understanding Score:</b> Macro-average score across Literature Understanding benchmarks.</div>
            <div class="tooltip-description-item"><b>Literature Understanding Cost:</b> Macro-average cost per problem (USD) across Literature Understanding benchmarks.</div>
            <div class="tooltip-description-item"><b>Code Execution Score:</b> Macro-average score across Code & Execution benchmarks.</div>
            <div class="tooltip-description-item"><b>Code Execution Cost:</b> Macro-average cost per problem (USD) across Code & Execution benchmarks.</div>
            <div class="tooltip-description-item"><b>Data Analysis Score:</b> Macro-average score across Data Analysis benchmarks.</div>
            <div class="tooltip-description-item"><b>Data Analysis Cost:</b> Macro-average cost per problem (USD) across Data Analysis benchmarks.</div>
            <div class="tooltip-description-item"><b>End-to-End Discovery Score:</b> Macro-average score across End-to-End Discovery benchmarks.</div>
            <div class="tooltip-description-item"><b>End-to-End Discovery Cost:</b> Macro-average cost per problem (USD)across End-to-End Discovery benchmarks.</div>
            <div class="tooltip-description-item"><b>Categories Attempted:</b> Number of core categories with at least one benchmark attempted (out of 4).</div>
            <div class="tooltip-description-item"><b>Logs:</b> View evaluation run logs (e.g., outputs, traces).</div>
        """
    elif table in ["Literature Understanding", "Code & Execution", "Data Analysis", "End-to-End Discovery"]:
        return f"""
            <div class="tooltip-description-item"><b>Agent:</b> Name of the evaluated agent.</div>
            <div class="tooltip-description-item"><b>Submitter:</b> Organization or individual who submitted the agent for evaluation.</div>
            <div class="tooltip-description-item"><b>LLM Base:</b> Model(s) used by the agent. Hover over ⓘ to view all.</div>
            <div class="tooltip-description-item"><b>{table} Score:</b> Macro-average score across {table} benchmarks.</div>
            <div class="tooltip-description-item"><b>{table} Cost:</b> Macro-average cost per problem (USD) across {table} benchmarks.</div>
            <div class="tooltip-description-item"><b>Benchmark Score:</b> Average (mean) score on the benchmark.</div>
            <div class="tooltip-description-item"><b>Benchmark Cost:</b> Average (mean) cost per problem (USD) on the benchmark.</div>
            <div class="tooltip-description-item"><b>Benchmarks Attempted:</b> Number of benchmarks attempted in this category (e.g., 3/5).</div>
            <div class="tooltip-description-item"><b>Logs:</b> View evaluation run logs (e.g., outputs, traces).</div>
        """
    else:
        # Fallback for any other table type, e.g., individual benchmarks
        return f"""
            <div class="tooltip-description-item"><b>Agent:</b> Name of the evaluated agent.</div>
            <div class="tooltip-description-item"><b>Submitter:</b> Organization or individual who submitted the agent for evaluation.</div>
            <div class="tooltip-description-item"><b>LLM Base:</b> Model(s) used by the agent. Hover over ⓘ to view all.</div>
            <div class="tooltip-description-item"><b>Benchmark Attempted:</b> Indicates whether the agent attempted this benchmark.</div>
            <div class="tooltip-description-item"><b>{table} Score:</b> Score achieved by the agent on this benchmark.</div>
            <div class="tooltip-description-item"><b>{table} Cost:</b> Cost incurred by the agent to solve this benchmark (in USD).</div>
            <div class="tooltip-description-item"><b>Logs:</b> View evaluation run logs (e.g., outputs, traces).</div>
        """

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

pareto_tooltip_content = build_pareto_tooltip_content()
openness_tooltip_content = build_openness_tooltip_content()
tooling_tooltip_content = build_tooling_tooltip_content()

def create_legend_markdown(which_table: str) -> str:
    """
    Generates the complete HTML for the legend section, including tooltips.
    This is used in the main leaderboard display.
    """
    descriptions_tooltip_content = build_descriptions_tooltip_content(which_table)
    legend_markdown = f"""
    <div style="display: flex; flex-wrap: wrap; align-items: flex-start; gap: 10px; font-size: 14px; padding-bottom: 8px;">
            
        <div> <!-- Container for the Pareto section -->
            <b>Pareto</b>
            <span class="tooltip-icon-legend">
                ⓘ
                <span class="tooltip-card">{pareto_tooltip_content}</span>
            </span>
            <div style="margin-top: 8px;"><span>🏆 On frontier</span></div>
        </div>
    
        <div> <!-- Container for the Openness section -->
            <b>Agent Openness</b>
            <span class="tooltip-icon-legend">
                ⓘ
                <span class="tooltip-card">
                    <h3>Agent Openness</h3>
                    <p class="tooltip-description">Indicates how transparent and reproducible an agent is.</p>
                    <div class="tooltip-items-container">{openness_tooltip_content}</div>
                </span>
            </span>
            <div style="display: flex; flex-wrap: wrap; align-items: center; gap: 16px; margin-top: 8px;">{openness_html}</div>
        </div>
    
        <div> <!-- Container for the Tooling section -->
            <b>Agent Tooling</b>
            <span class="tooltip-icon-legend">
                ⓘ
                <span class="tooltip-card">
                    <h3>Agent Tooling</h3>
                    <p class="tooltip-description">Describes the tool usage and execution environment of the agent during evaluation.</p>
                    <div class="tooltip-items-container">{tooling_tooltip_content}</div>
                </span>
            </span>
            <div style="display: flex; flex-wrap: wrap; align-items: center; gap: 16px; margin-top: 8px;">{tooling_html}</div>
        </div>
        
        <div><!-- Container for the Column Descriptions section -->
            <b>Column Descriptions</b>
            <span class="tooltip-icon-legend">
                ⓘ
                <span class="tooltip-card">
                    <h3>Column Descriptions</h3>
                    <div class="tooltip-items-container">{descriptions_tooltip_content}</div>
                </span>
            </span>
        </div>
    </div>
    """
    return legend_markdown

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
    columns_to_drop = ['id', 'Openness', 'Agent Tooling']
    df_view = df_view.drop(columns=columns_to_drop, errors='ignore')

    df_headers = df_view.columns.tolist()
    df_datatypes = []
    for col in df_headers:
        if col == "Logs" or "Cost" in col or "Score" in col:
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
    # Dynamically set widths for the DataFrame columns
    fixed_start_widths = [40, 40, 200, 150, 200, 120, 120]
    num_score_cost_cols = 0
    remaining_headers = df_headers[len(fixed_start_widths):]
    for col in remaining_headers:
        if "Score" in col or "Cost" in col:
            num_score_cost_cols += 1
    dynamic_widths = [80] * num_score_cost_cols
    fixed_end_widths = [80, 40]
    # 5. Combine all the lists to create the final, fully dynamic list.
    final_column_widths = fixed_start_widths + dynamic_widths + fixed_end_widths

    plot_component = gr.Plot(
        value=scatter_plot,
        show_label=False
    )
    gr.HTML(value=scatter_disclaimer_html, elem_id="scatter-disclaimer")
    # Put table and key into an accordion
    with gr.Accordion("Show / Hide Table View", open=True, elem_id="leaderboard-accordion"):
        dataframe_component = gr.DataFrame(
            headers=df_headers,
            value=df_view,
            datatype=df_datatypes,
            interactive=False,
            wrap=True,
            column_widths=final_column_widths,
            elem_classes=["wrap-header-df"]
        )
        legend_markdown = create_legend_markdown(category_name)
        gr.HTML(value=legend_markdown, elem_id="legend-markdown")

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
            gr.DataFrame(
                headers=df_headers,
                value=benchmark_table_df,
                datatype=df_datatypes,
                interactive=False,
                wrap=True,
                column_widths=[40, 40, 200, 150, 175, 85, 100, 100, 40],
                elem_classes=["wrap-header-df"]
            )
            legend_markdown = create_legend_markdown(benchmark_name)
            gr.HTML(value=legend_markdown, elem_id="legend-markdown")

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
