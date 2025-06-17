import matplotlib
matplotlib.use('Agg')

import pandas as pd
import numpy as np # For mock data generation
import plotly.express as px
import plotly.graph_objects as go

import json
import os
import shutil
import tarfile
import tempfile
from datetime import datetime, timedelta, timezone
from email.utils import parseaddr
from pathlib import Path
# from zoneinfo import ZoneInfo # LeaderboardViewer uses this, ensure it's available

import gradio as gr
import requests
from agenteval import (
    # compute_summary_statistics, # This will now be used by LeaderboardViewer
    process_eval_logs,
    upload_folder_to_hf,
    upload_summary_to_hf,
)
from agenteval.models import EvalResult # Used by submission and LeaderboardViewer (implicitly)
from agenteval.upload import sanitize_path_component
from datasets import Dataset, DatasetDict, VerificationMode, load_dataset # load_dataset used by LV
from datasets.data_files import EmptyDatasetError
from huggingface_hub import HfApi

from leaderboard_viewer import LeaderboardViewer
from json_leaderboard import LeaderboardViewer2

from content import (
    CITATION_BUTTON_LABEL,
    CITATION_BUTTON_TEXT,
    INTRODUCTION_TEXT,
    SUBMISSION_TEXT,
    INTRO_PARAGRAPH,
    format_error,
    format_log,
    format_warning,
    hf_uri_to_web_url, # LeaderboardViewer might need this or similar for log links
    hyperlink, # LeaderboardViewer might need this or similar for log links
)

# --- Constants and Configuration  ---
USE_MOCK_DATA = False
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

# --- Mock Data Generation Functions (if USE_MOCK_DATA is True) ---
# def generate_mock_dataframe(split_name: str, tag: str | None):
#     num_rows = 5
#     agents = [f"MockAgent-{split_name}-{i}" for i in range(1, num_rows + 1)]
#     data = {
#         "Agent": agents,
#         "User/organization": [f"Org{i % 2 + 1}" for i in range(num_rows)],
#         "Submission date": [f"2023-10-{10 + i}" for i in range(num_rows)],
#         "Logs": [f"[Logs](http://example.com/logs/{agent})" for agent in agents], # Mock links
#     }
#
#     primary_metric = tag if tag and tag != "Overall" else "Overall"
#
#     data[primary_metric] = np.random.rand(num_rows) * 80 + 10 # Scores between 10 and 90
#     data[f"{primary_metric} cost"] = np.random.rand(num_rows) * 50 + 5
#     if np.random.rand() > 0.3: # Sometimes include CIs
#         data[f"{primary_metric} 95% CI"] = np.random.rand(num_rows) * 5
#
#     if tag and tag != "Overall":
#         # Add some mock task scores under the tag
#         data[f"task_A_under_{tag}"] = np.random.rand(num_rows) * 70 + 15
#         data[f"task_A_under_{tag} cost"] = np.random.rand(num_rows) * 20 + 2
#         data[f"task_B_under_{tag}"] = np.random.rand(num_rows) * 60 + 25
#         data[f"task_B_under_{tag} cost"] = np.random.rand(num_rows) * 30 + 3
#     elif tag is None or tag == "Overall": # Add some general mock task/tag scores
#         data["Literature Understanding"] = np.random.rand(num_rows) * 75 + 10
#         data["MockTag1 cost"] = np.random.rand(num_rows) * 40 + 5
#         data["mock_task_X"] = np.random.rand(num_rows) * 65 + 20
#         data["mock_task_X cost"] = np.random.rand(num_rows) * 35 + 4
#
#
#     df = pd.DataFrame(data)
#     # Round numeric columns to 2 decimal points
#     numeric_cols = df.select_dtypes(include=["float", "int"]).columns
#     df[numeric_cols] = df[numeric_cols].round(2)
#     # Sort by primary metric
#     if primary_metric in df.columns:
#         df = df.sort_values(by=primary_metric, ascending=False).reset_index(drop=True)
#     return df
def generate_mock_dataframe(split_name: str, tag: str | None):
    """
    Generate a DataFrame from the agenteval.json file.
    """
    json_path = os.path.join(DATA_DIR, AGENTEVAL_MANIFEST_NAME)  # Path to agenteval.json

    try:
        # Load JSON data
        with open(json_path, 'r') as f:
            data = json.load(f)

        # Validate the structure of the JSON data
        if not isinstance(data, dict):
            raise ValueError("Invalid JSON structure: Expected a dictionary at the root.")

        # Extract relevant fields
        rows = []
        # print(f"OOOGA BOOOGA{data.get('results')}")
        for entry in data.get("results", []):  # Ensure "results" is a key in the JSON
            # if not isinstance(entry, list):
            #     raise ValueError("Invalid JSON structure: Expected 'results' to contain a list.")
            for model_data in entry:
                if not isinstance(model_data, dict):
                    raise ValueError("Invalid JSON structure: Expected model data to be a dictionary.")
                model = model_data.get("model", "Unknown Model")
                usage = model_data.get("usage", {})
                rows.append({
                    "Model": model,
                    "Input Tokens": usage.get("input_tokens", 0),
                    "Output Tokens": usage.get("output_tokens", 0),
                    "Total Tokens": usage.get("total_tokens", 0),
                    "Cache Read Tokens": usage.get("input_tokens_cache_read", 0),
                    "Cache Write Tokens": usage.get("input_tokens_cache_write", 0),
                    "Reasoning Tokens": usage.get("reasoning_tokens", 0),
                })

        # Create DataFrame
        df = pd.DataFrame(rows)

        # Add split and tag columns for context
        df["splits"] = split_name
        df["Tags"] = tag if tag else "Overall"

        # Round numeric columns to 2 decimal points
        numeric_cols = df.select_dtypes(include=["float", "int"]).columns
        df[numeric_cols] = df[numeric_cols].round(2)

        # Sort by a primary metric (e.g., Total Tokens)
        if "Total Tokens" in df.columns:
            df = df.sort_values(by="Total Tokens", ascending=False).reset_index(drop=True)

        return df

    except Exception as e:
        print(f"Error generating DataFrame from JSON: {e}")
        return pd.DataFrame()  # Return an empty DataFrame on error

# def generate_mock_scatter_plot(df: pd.DataFrame, score_col: str, cost_col: str, agent_name_col: str = "Agent"):
#     if not isinstance(df, pd.DataFrame):
#         # Return an empty figure or a message if df is not a DataFrame
#         fig = go.Figure()
#         fig.update_layout(title_text="Error: Input data is not a valid DataFrame.",
#                           xaxis_visible=False, yaxis_visible=False,
#                           annotations=[dict(text="No data to display.", showarrow=False)])
#         return fig
#
#     if score_col not in df.columns or cost_col not in df.columns:
#         fig = go.Figure()
#         fig.update_layout(title_text=f"Missing Data for Plot",
#                           xaxis_title=cost_col, yaxis_title=score_col,
#                           annotations=[dict(text=f"Data for '{score_col}' or '{cost_col}' is missing.", showarrow=False)])
#         return fig
#
#     plot_data = df.copy()
#     plot_data[score_col] = pd.to_numeric(plot_data[score_col], errors='coerce')
#     plot_data[cost_col] = pd.to_numeric(plot_data[cost_col], errors='coerce')
#
#     # Ensure agent_name_col exists, otherwise hover might fail or show 'None'
#     if agent_name_col not in plot_data.columns:
#         # Create a placeholder if it doesn't exist to avoid errors,
#         # or handle this more gracefully based on your data structure.
#         plot_data[agent_name_col] = "Unknown Agent"
#
#
#     # Create the main scatter plot
#     fig = px.scatter(
#         plot_data,
#         x=cost_col,
#         y=score_col,
#         color=agent_name_col, # Color by agent
#         hover_name=agent_name_col, # This will be the bold title in the hover
#         hover_data={ # Additional data to show on hover
#             score_col: ':.2f', # Format score to 2 decimal places
#             cost_col: ':.2f',  # Format cost to 2 decimal places
#             # Add any other columns you want in the hover tooltip
#             # 'Original_Agent_ID': True # if you have another ID column
#         },
#         title=f"Interactive: {score_col} vs. {cost_col}",
#         labels={cost_col: f"{cost_col} (USD)", score_col: f"{score_col} Score"}
#     )
#
#     fig.update_traces(marker=dict(size=10), selector=dict(mode='markers'))
#     fig.update_layout(xaxis_rangemode='tozero', yaxis_rangemode='tozero') # Similar to xlim(left=0), ylim(bottom=0)
#
#     # --- Pareto Frontier Calculation and Plotting ---
#     frontier_data = plot_data.dropna(subset=[score_col, cost_col])
#     pareto_points_mock = []
#     if not frontier_data.empty:
#         frontier_data = frontier_data.sort_values(by=[cost_col, score_col], ascending=[True, False])
#         max_score_at_cost_mock = -np.inf
#         for index, row in frontier_data.iterrows():
#             # Ensure score is numeric and not NaN for comparison
#             current_score = pd.to_numeric(row[score_col], errors='coerce')
#             if pd.notna(current_score) and current_score > max_score_at_cost_mock:
#                 pareto_points_mock.append(row)
#                 max_score_at_cost_mock = current_score
#
#         if pareto_points_mock:
#             pareto_df_mock = pd.DataFrame(pareto_points_mock).sort_values(by=cost_col)
#             if not pareto_df_mock.empty:
#                 fig.add_trace(go.Scatter(
#                     x=pareto_df_mock[cost_col],
#                     y=pareto_df_mock[score_col],
#                     mode='lines+markers',
#                     name='Pareto Frontier',
#                     line=dict(color='red', width=2),
#                     marker=dict(size=5)
#                 ))
#
#     fig.update_layout(
#         legend_title_text="Agent / Frontier",
#         legend=dict(
#             orientation="v",
#             yanchor="top",
#             y=1.02,
#             xanchor="left",
#             x=1.02
#         ),
#         margin=dict(r=200) # Add right margin to make space for legend
#     )
#
#     return fig
#
#
#
# def get_mock_tag_map(split_name: str):
#     # Simulate the tag_map structure LeaderboardViewer would provide
#     return {
#         "Overall": [], # Overall usually doesn't list tasks under it in this way
#         "Literature Understanding": [f"mock_task_A_under_MockTag1", f"mock_task_B_under_MockTag1"],
#         f"E2E_{split_name}": [f"task_X_under_SpecificTag_{split_name}", f"task_Y_under_SpecificTag_{split_name}"]
#     }

# --- Global State for Viewers (simple caching) ---
CACHED_VIEWERS = {}
CACHED_TAG_MAPS = {} # Will now also be populated by mock tag map

def get_leaderboard_viewer_instance(split: str):
    global CACHED_VIEWERS, CACHED_TAG_MAPS

    if USE_MOCK_DATA:
        print(f"MOCK MODE: Generating mock data for split: {split}")
        # For mock mode, we don't cache a "viewer" but the result of "view()"
        # The "viewer" in this case is just the concept of providing data.
        # The tag map is also mocked.
        mock_tag_map = get_mock_tag_map(split)
        CACHED_TAG_MAPS[split] = mock_tag_map # Cache the mock tag map
        # Return a "mock viewer concept" and its tag map.
        # The actual data generation will happen in load_display_data_for_split when this mock is detected.
        return "MOCK_VIEWER_PLACEHOLDER", mock_tag_map

    # --- Real LeaderboardViewer logic
    TARGET_JSON_PATH = "data/1.0.0-dev1/agenteval.json"
    if split in CACHED_VIEWERS:
        return CACHED_VIEWERS[split], CACHED_TAG_MAPS.get(split, {"Overall": []})

    if LeaderboardViewer is None:
        raise ImportError("LeaderboardViewer class is not available.")

    print(f"Initializing LeaderboardViewer for split: {split}")
    try:
        if TARGET_JSON_PATH:
            print(f"Using direct JSON for split '{split}': {TARGET_JSON_PATH}")
            viewer = LeaderboardViewer2(
                json_file_path=TARGET_JSON_PATH,
                split=split,   # Still needed for context within the JSON's suite_config
                is_internal=IS_INTERNAL
            )
        else:
            print(f"Using Hugging Face dataset for split '{split}': {RESULTS_DATASET}/{CONFIG_NAME}")
            viewer = LeaderboardViewer(
                repo_id=RESULTS_DATASET,
                config=CONFIG_NAME,
                split=split,
                is_internal=IS_INTERNAL
            )
        current_tag_map = {"Overall": []} # Start with Overall
        current_tag_map.update({k: v for k, v in viewer.tag_map.items()}) # Add tags from viewer
        CACHED_TAG_MAPS[split] = current_tag_map

        CACHED_VIEWERS[split] = viewer
        return viewer, CACHED_TAG_MAPS[split]
    except ValueError as e:
        print(format_warning(f"Could not initialize LeaderboardViewer for split '{split}': {e}"))
        dummy_df = pd.DataFrame({"Message": [f"No data or error for split '{split}'."]})
        dummy_plots = {}
        CACHED_VIEWERS[split] = (dummy_df, dummy_plots)
        CACHED_TAG_MAPS[split] = {"Overall": []} # Default tag map on error
        return (dummy_df, dummy_plots), CACHED_TAG_MAPS[split]
    except Exception as e:
        print(format_error(f"Unexpected error initializing LeaderboardViewer for split '{split}': {e}"))
        dummy_df = pd.DataFrame({"Message": [f"Error loading data for split '{split}'."]})
        dummy_plots = {}
        CACHED_VIEWERS[split] = (dummy_df, dummy_plots)
        CACHED_TAG_MAPS[split] = {"Overall": []} # Default tag map on error
        return (dummy_df, dummy_plots), CACHED_TAG_MAPS[split]


def load_display_data_for_split(split: str, selected_tag: str | None):
    global USE_MOCK_DATA
    json_path = os.path.join(DATA_DIR, AGENTEVAL_MANIFEST_NAME)
    viewer_or_data, tag_map_for_split = get_leaderboard_viewer_instance(split)

    primary_metric_name_for_plot = selected_tag if selected_tag and selected_tag != "Overall" else "Overall"

    actual_tag_for_view = selected_tag if selected_tag and selected_tag != "Overall" else None

    df = pd.DataFrame() # Initialize df
    scatter_plot = None # Initialize scatter_plot

    if USE_MOCK_DATA and viewer_or_data == "MOCK_VIEWER_PLACEHOLDER":
        print(f"MOCK MODE: Serving mock view for split: {split}, tag: {actual_tag_for_view}")
        df = generate_mock_dataframe(split, actual_tag_for_view)

        # Define score and cost columns for the mock scatter plot based on primary_metric_name_for_plot
        mock_score_col = primary_metric_name_for_plot
        mock_cost_col = f"{primary_metric_name_for_plot} cost"
        scatter_plot = generate_mock_scatter_plot(df, score_col=mock_score_col, cost_col=mock_cost_col)

    elif isinstance(viewer_or_data, LeaderboardViewer2): # Real viewer
        viewer = viewer_or_data
        # It returns a dict `plots_dict`
        df, plots_dict = viewer.view(tag=actual_tag_for_view, with_plots=True, use_plotly=True)
        # Extract the correct scatter plot based on primary_metric_name_for_plot
        # The keys in plots_dict are like "scatter_{primary_metric_name_for_plot}"
        scatter_plot_key = f"scatter_{primary_metric_name_for_plot}"
        scatter_plot = plots_dict.get(scatter_plot_key)
        if not scatter_plot: # Fallback if specific scatter plot not found by that exact key
            for key, value in plots_dict.items():
                if key.startswith("scatter_"): # Pick the first available scatter
                    scatter_plot = value
                    print(f"Warning: Scatter plot for '{primary_metric_name_for_plot}' not found, using fallback: {key}")
                    break


    elif isinstance(viewer_or_data, tuple): # Pre-processed dummy data from error in real viewer init
        df, _ = viewer_or_data # plots_dict might be empty or contain dummy bar
        # If we have a df, we can try to make a generic mock scatter plot
        if not df.empty:
            scatter_plot = generate_mock_scatter_plot(df, score_col="Overall", cost_col="Overall cost") # Generic attempt

    else: # Should not happen
        df = pd.DataFrame({"Error": ["Unexpected state in get_leaderboard_viewer_instance"]})
        # scatter_plot remains None

    # --- Post-processing (common for real and mock data, if needed) ---
    if "Logs" in df.columns and not (USE_MOCK_DATA and "Logs" in df.columns and df["Logs"].str.startswith("[").any()): # Avoid re-processing mock logs
        try:
            df["Logs"] = df["Logs"].apply(
                lambda x: hyperlink(hf_uri_to_web_url(x), "🔗") if isinstance(x, str) and x.startswith("datasets/") else (hyperlink(x, "🔗") if isinstance(x, str) and x.startswith("http") else x)
            )
        except Exception as e:
            print(f"Could not convert Logs to Markdown for split {split}: {e}")
    df = df.fillna("")

    tag_choices_for_dropdown = ["Overall"] + sorted([k for k in tag_map_for_split.keys() if k != "Overall"])
    tag_choices_for_dropdown = list(dict.fromkeys(tag_choices_for_dropdown))

    # Return only DataFrame, Scatter Plot, and Dropdown Update
    return df, scatter_plot, gr.update(choices=tag_choices_for_dropdown, value=selected_tag)


# --- Helper to get initial tags for the global dropdown ---
def get_initial_global_tag_choices():
    global USE_MOCK_DATA
    initial_split_for_tags = "validation" if IS_INTERNAL else "test"
    try:
        # This will now correctly use the mock tag map if USE_MOCK_DATA is True
        _, tag_map_for_initial_split = get_leaderboard_viewer_instance(initial_split_for_tags)

        if isinstance(tag_map_for_initial_split, dict):
            choices = sorted([k for k in tag_map_for_initial_split.keys() if k != "Overall"])
            choices.insert(0, "Overall")
            return list(dict.fromkeys(choices))
        else:
            print(f"Warning: Tag map for initial split '{initial_split_for_tags}' was not a dict. Defaulting tags.")
            return ["Overall"]
    except Exception as e:
        print(f"Error getting initial global tag choices for split '{initial_split_for_tags}': {e}. Defaulting tags.")
        return ["Overall"]



# --- Submission Logic (largely unchanged from original, ensure EvalResult and other deps are fine) ---
def try_load_dataset_submission(*args, **kwargs) -> DatasetDict: # Renamed to avoid conflict if LV has one
    try:
        return load_dataset(*args, **kwargs)
    except EmptyDatasetError:
        return DatasetDict()
    except ValueError: # Handles cases where dataset is empty or ill-formed
        return DatasetDict()

def checked_upload_folder(
        api_hf: HfApi, # Renamed to avoid conflict with global api
        folder_path: str,
        repo_id: str,
        config_name_ul: str, # Renamed
        split_ul: str, # Renamed
        submission_name_ul: str, # Renamed
) -> str:
    total = 0
    for root, _, files in os.walk(folder_path):
        for f_ul in files: # Renamed
            total += os.path.getsize(os.path.join(root, f_ul))
            if total > MAX_UPLOAD_BYTES:
                raise ValueError(
                    f"Upload too large: exceeds {MAX_UPLOAD_BYTES // (1024**2)} MB limit."
                )
    return upload_folder_to_hf(
        api=api_hf, # Use renamed parameter
        folder_path=folder_path,
        repo_id=repo_id,
        config_name=config_name_ul,
        split=split_ul,
        submission_name=submission_name_ul,
    )

def add_new_eval(
        val_or_test: str,
        agent_name: str | None,
        agent_description: str,
        agent_url: str,
        path_to_file: tempfile._TemporaryFileWrapper | None,
        username: str,
        mail: str,
        profile: gr.OAuthProfile,
        # We need global eval_results for checks; this might need rethinking if it's purely display driven now
        # For now, let's assume we still load it for submission checks
):
    # Load current eval_results for submission checks
    # This is a bit redundant if display part reloads it, but submission needs its own consistent view
    current_eval_results_for_submission = try_load_dataset_submission(
        RESULTS_DATASET,
        CONFIG_NAME,
        download_mode="force_redownload", # Or a less aggressive mode
        verification_mode=VerificationMode.NO_CHECKS,
        trust_remote_code=True,
    )
    if not agent_name:
        return format_warning("Please provide an agent name.")

    submission_time = datetime.now(timezone.utc)
    if not username or username.strip() == "":
        username = profile.username # Default to HF username

    # User account age check
    try:
        user_data_resp = requests.get(f"https://huggingface.co/api/users/{profile.username}/overview")
        user_data_resp.raise_for_status()
        creation_date_str = user_data_resp.json()["createdAt"]
        created_at = datetime.strptime(creation_date_str, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone.utc)
        if submission_time - created_at < timedelta(days=60):
            return format_error("This account is not authorized to submit here (account too new).")
    except Exception as e:
        print(f"Error checking user account age: {e}")
        return format_error("Could not verify account age. Please try again later.")

    # Submission frequency check
    contact_infos = try_load_dataset_submission(
        CONTACT_DATASET, CONFIG_NAME, download_mode="force_redownload",
        verification_mode=VerificationMode.NO_CHECKS, trust_remote_code=True
    )
    user_submission_dates = sorted(
        datetime.fromisoformat(row["submit_time"])
        for row in contact_infos.get(val_or_test, []) if row["username_auth"] == profile.username
    )
    if user_submission_dates and (submission_time - user_submission_dates[-1] < timedelta(days=1)):
        return format_error("You already submitted once in the last 24h for this split; please try again later.")

    # Email validation
    _, parsed_mail = parseaddr(mail)
    if "@" not in parsed_mail:
        return format_warning("Please provide a valid email address.")

    # Duplicate submission check
    if val_or_test in current_eval_results_for_submission and len(current_eval_results_for_submission[val_or_test]) > 0:
        existing_submissions = current_eval_results_for_submission[val_or_test].to_dict().get("submission", [])
        for sub_item in existing_submissions:
            if (sub_item.get("agent_name", "").lower() == agent_name.lower() and
                    sub_item.get("username", "").lower() == username.lower()):
                return format_warning("This agent name by this user has already been submitted to this split.")

    if path_to_file is None:
        return format_warning("Please attach a .tar.gz file.")

    safe_username = sanitize_path_component(username)
    safe_agent_name = sanitize_path_component(agent_name)
    extracted_dir = os.path.join(EXTRACTED_DATA_DIR, f"{safe_username}_{safe_agent_name}")

    # File extraction
    if not LOCAL_DEBUG:
        try:
            if os.path.exists(extracted_dir): shutil.rmtree(extracted_dir)
            os.makedirs(extracted_dir, exist_ok=True)
            with tarfile.open(path_to_file.name, "r:gz") as tar:
                members_extracted = 0
                for member in tar.getmembers():
                    if not member.isreg(): continue
                    fname = os.path.basename(member.name)
                    if not fname or fname.startswith("."): continue
                    fobj = tar.extractfile(member)
                    if not fobj: continue
                    with open(os.path.join(extracted_dir, fname), "wb") as out:
                        out.write(fobj.read())
                    members_extracted +=1
                if members_extracted == 0:
                    return format_error("Submission tarball is empty or contains no valid files.")
        except Exception as e:
            return format_error(f"Error extracting file: {e}. Ensure it's a valid .tar.gz.")
    else: print("mock extracted file", flush=True)


    submission_name = f"{safe_username}_{safe_agent_name}_{submission_time.strftime('%Y-%m-%d_%H-%M-%S')}"

    # 1. Upload raw (unscored) submission files
    if not LOCAL_DEBUG:
        try:
            checked_upload_folder(api, extracted_dir, SUBMISSION_DATASET, CONFIG_NAME, val_or_test, submission_name)
        except ValueError as e: return format_error(str(e))
        except Exception as e: return format_error(f"Failed to upload raw submission: {e}")
    else: print("mock uploaded raw submission", flush=True)

    # 2. Save contact information
    contact_info = {
        "agent_name": agent_name, "agent_description": agent_description, "url": agent_url,
        "username": username, "username_auth": profile.username, "mail": mail,
        "submit_time": submission_time.isoformat(),
    }
    if val_or_test in contact_infos:
        contact_infos[val_or_test] = contact_infos[val_or_test].add_item(contact_info)
    else:
        contact_infos[val_or_test] = Dataset.from_list([contact_info])

    if not LOCAL_DEBUG:
        try:
            contact_infos.push_to_hub(CONTACT_DATASET, config_name=CONFIG_NAME)
        except Exception as e: return format_warning(f"Submission recorded, but contact info failed to save: {e}")
    else: print("mock uploaded contact info", flush=True)


    # 3. Process and score the submission
    eval_result_obj = None # Define to avoid NameError
    try:
        json_path = Path(extracted_dir) / AGENTEVAL_MANIFEST_NAME
        if not json_path.exists():
            return format_error(f"Missing manifest {AGENTEVAL_MANIFEST_NAME} in submission.")

        eval_result_obj = EvalResult.model_validate_json(json_path.read_text(encoding="utf-8"))
        if eval_result_obj.suite_config.version != CONFIG_NAME:
            return format_error(f"Suite version mismatch: expected {CONFIG_NAME}, got {eval_result_obj.suite_config.version}.")
        if eval_result_obj.split != val_or_test:
            return format_error(f"Split mismatch: expected {val_or_test}, got {eval_result_obj.split}.")

        # Re-compute results from logs for integrity
        eval_result_obj.results = process_eval_logs(extracted_dir)[0] # Assuming process_eval_logs returns a tuple/list
        eval_result_obj.save_json(str(json_path)) # Save the re-processed manifest

    except Exception as e:
        return format_error(f"Error scoring submission: {e}. Check manifest and log files.")

    # 4. Upload scored submission files
    logs_url_private_val, logs_url_public_val = None, None
    scored_submission_name = f"{submission_name}_scored"
    if not LOCAL_DEBUG:
        try:
            logs_url_private_val = checked_upload_folder(api, extracted_dir, SUBMISSION_DATASET, CONFIG_NAME, val_or_test, scored_submission_name)
            if val_or_test == "validation" and not IS_INTERNAL: # Public copy for validation
                logs_url_public_val = checked_upload_folder(api, extracted_dir, SUBMISSION_DATASET_PUBLIC, CONFIG_NAME, val_or_test, scored_submission_name)
        except ValueError as e: return format_error(str(e))
        except Exception as e: return format_error(f"Failed to upload scored submission: {e}")
    else: print("mock uploaded scored submission", flush=True)


    # Update EvalResult with submission details
    eval_result_obj.submission.agent_name = agent_name
    eval_result_obj.submission.agent_description = agent_description
    eval_result_obj.submission.agent_url = agent_url
    eval_result_obj.submission.username = username
    eval_result_obj.submission.submit_time = submission_time
    eval_result_obj.submission.logs_url = logs_url_private_val
    eval_result_obj.submission.logs_url_public = logs_url_public_val

    # 5. Upload summary statistics to RESULTS_DATASET (for the leaderboard)
    if not LOCAL_DEBUG:
        try:
            upload_summary_to_hf(api, eval_result_obj, RESULTS_DATASET, CONFIG_NAME, val_or_test, scored_submission_name)
        except Exception as e:
            return format_error(f"Failed to upload summary results to leaderboard: {e}")
    else: print("mock uploaded results to lb", flush=True)

    # Invalidate viewer cache for the split that was updated
    if val_or_test in CACHED_VIEWERS:
        del CACHED_VIEWERS[val_or_test]
    if val_or_test in CACHED_TAG_MAPS:
        del CACHED_TAG_MAPS[val_or_test]


    return format_log(
        f"Agent '{agent_name}' submitted successfully by '{username}' to '{val_or_test}' split. "
        "Please refresh the leaderboard in a few moments. It may take some time for changes to propagate."
    )

with gr.Blocks() as demo:
    gr.Markdown(INTRODUCTION_TEXT, elem_classes="markdown-text")
    gr.HTML(INTRO_PARAGRAPH, elem_id="intro-paragraph")

    # --- Submission Accordion ---
    with gr.Accordion("🚀 Submit a new agent for evaluation", open=False, elem_classes="submission-accordion"):
        gr.Markdown(SUBMISSION_TEXT, elem_id="markdown-text")
        with gr.Row():
            with gr.Column():
                level_of_test_radio = gr.Radio(["validation", "test"], value="validation", label="Split")
                agent_name_tb = gr.Textbox(label="Agent Name")
                agent_desc_tb = gr.Textbox(label="Agent Description")
                agent_url_tb = gr.Textbox(label="URL to Agent Information")
            with gr.Column():
                username_tb = gr.Textbox(label="Organization or User Name (Defaults to HF username)")
                mail_tb = gr.Textbox(label="Contact Email (Private, for submission issues)")
                file_upload_comp = gr.File(
                    label="Submission File (.tar.gz ...)", # Shortened for brevity
                    file_types=[".gz", ".tar.gz"]
                )
        with gr.Row():
            gr.LoginButton()
            submit_eval_button = gr.Button("Submit Evaluation")
        submission_result = gr.Markdown()

        submit_eval_button.click(
            add_new_eval,
            [
                level_of_test_radio,
                agent_name_tb,
                agent_desc_tb,
                agent_url_tb,
                file_upload_comp,
                username_tb,
                mail_tb
            ],
            submission_result,
        )

    # --- Leaderboard Display Section (within main_page) ---
    gr.Markdown("---") # Separator
    gr.Markdown("## Leaderboard Results")

    # Shared controls for plots and tag selection
    with gr.Row():
        # Assuming get_initial_global_tag_choices is defined elsewhere
        # initial_choices_for_dropdown = get_initial_global_tag_choices()
        initial_choices_for_dropdown = ["Overall", "Literature Understanding"] # Placeholder if function not available yet
        global_tag_dropdown = gr.Dropdown(
            choices=initial_choices_for_dropdown,
            value="Overall",
            label="Filter by Tag/Capability",
            elem_id="global_tag_dropdown" # Note: elem_id should be unique if multiple instances exist
        )

    # Tabs for Test and Validation
    with gr.Tab("Results: Validation") as val_tab:
        val_split_name = gr.State("validation")
        with gr.Row():
            val_refresh_button = gr.Button("Refresh Validation Tab")
        val_df_output = gr.DataFrame(interactive=False, wrap=True, label="Validation Leaderboard")
        val_scatter_plot_output = gr.Plot(label="Score vs. Cost (Validation)")

        initial_val_df, initial_val_scatter, _ = load_display_data_for_split(
            "validation", global_tag_dropdown.value
        )
        val_df_output.value = initial_val_df
        val_scatter_plot_output.value = initial_val_scatter

        val_refresh_button.click(
            fn=load_display_data_for_split,
            inputs=[val_split_name, global_tag_dropdown],
            outputs=[val_df_output, val_scatter_plot_output, global_tag_dropdown]
        )

    with gr.Tab("Results: Test") as test_tab:
        test_split_name = gr.State("test")
        with gr.Row():
            test_refresh_button = gr.Button("Refresh Test Tab")
        test_df_output = gr.DataFrame(interactive=False, wrap=True, label="Test Leaderboard")

        test_scatter_plot_output = gr.Plot(label="Score vs. Cost (Test)")


        initial_test_df, initial_test_scatter, _ = load_display_data_for_split(
            "test", global_tag_dropdown.value
        )
        test_df_output.value = initial_test_df
        test_scatter_plot_output.value = initial_test_scatter

        test_refresh_button.click(
            fn=load_display_data_for_split,
            inputs=[test_split_name, global_tag_dropdown],
            outputs=[test_df_output, test_scatter_plot_output, global_tag_dropdown]
        )

    with gr.Accordion("📙 Citation", open=False):
        gr.Textbox(value=CITATION_BUTTON_TEXT, label=CITATION_BUTTON_LABEL, elem_id="citation-button-main", interactive=False)


if __name__ == "__main__":
    demo.launch()