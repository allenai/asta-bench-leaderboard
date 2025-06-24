import matplotlib
matplotlib.use('Agg')

import pandas as pd
import plotly.graph_objects as go

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
from agenteval.leaderboard.upload import sanitize_path_component
from agenteval.leaderboard.view import LeaderboardViewer
from datasets import Dataset, DatasetDict, VerificationMode, load_dataset # load_dataset used by LV
from datasets.data_files import EmptyDatasetError
from huggingface_hub import HfApi

import leaderboard_transformer
from leaderboard_transformer import DataTransformer, create_pretty_tag_map, INFORMAL_TO_FORMAL_NAME_MAP, transform_raw_dataframe

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


def load_display_data_for_split(split: str, selected_tag: str | None):
    """
    Loads, transforms, and prepares all data and plots for a given split and tag.
    This is the single source of truth for updating the UI.
    """
    # Default to "Overall" if the tag is None or empty
    if not selected_tag:
        selected_tag = "Overall"

    viewer_or_data, raw_tag_map = get_leaderboard_viewer_instance(split)

    # Initialize with empty/default values
    df = pd.DataFrame({"Message": [f"No data available for split '{split}'."]})
    scatter_plot = go.Figure()
    tag_choices = [("Overall", "Overall")]

    if isinstance(viewer_or_data, LeaderboardViewer):
        # --- The Clean Workflow ---
        raw_df, _ = viewer_or_data._load()
        pretty_df = transform_raw_dataframe(raw_df)
        pretty_tag_map = create_pretty_tag_map(raw_tag_map, INFORMAL_TO_FORMAL_NAME_MAP)

        transformer = DataTransformer(pretty_df, pretty_tag_map)
        df, plots_dict = transformer.view(tag=selected_tag, use_plotly=True)

        # --- Simplified Plot Retrieval ---
        # We now get the plot with a consistent key, no more guessing!
        scatter_plot = plots_dict.get('scatter_plot', go.Figure())

        # Update the dropdown choices from the pretty map
        tag_choices = [("Overall", "Overall")] + sorted([(k, k) for k in pretty_tag_map.keys()])

    # --- Post-processing (remains the same) ---
    if "Logs" in df.columns:
        def format_log_entry_to_html(raw_uri):
            if pd.isna(raw_uri) or raw_uri == "": # Handle None, NaN, or empty strings
                return "" # No link if URI is missing

            web_url = hf_uri_to_web_url(str(raw_uri)) # Ensure it's a string

            if web_url: # Only create hyperlink if we have a valid web_url
                return hyperlink(web_url, "🔗")
            return "" # Return empty if web_url couldn't be formed
        df["Logs"] = df["Logs"].apply(format_log_entry_to_html)
    df = df.fillna("")

    # The order of this return statement is CRITICAL for Gradio event handlers
    return df, scatter_plot, gr.update(choices=tag_choices, value=selected_tag)


def create_leaderboard_tab(split_name: str, global_dropdown: gr.Dropdown):
    """
    A helper function to create the UI components for a single leaderboard tab.
    This avoids duplicating code for the 'validation' and 'test' tabs.
    """
    # --- 1. Initial Data Load for this Tab ---
    # This runs ONCE when the app starts to populate the initial view.
    initial_df, initial_scatter, initial_tags_update = load_display_data_for_split(
        split_name, global_dropdown.value
    )

    # Update the global dropdown's choices based on the first tab that loads.
    global_dropdown.choices = initial_tags_update['choices']

    # --- 2. Define UI Components for this Tab ---
    with gr.Row():
        refresh_button = gr.Button(f"Refresh {split_name.capitalize()} Tab")

    # Define headers and datatypes based on the initial DataFrame
    df_headers = initial_df.columns.tolist()
    df_datatypes = ["markdown" if col == "Logs" else "number" for col in df_headers]

    df_output = gr.DataFrame(
        headers=df_headers,
        value=initial_df,
        datatype=df_datatypes,
        interactive=False,
        wrap=True,
        label=f"{split_name.capitalize()} Leaderboard"
    )

    # The plot is now correctly populated on initial load via its `value`
    scatter_plot_output = gr.Plot(
        value=initial_scatter,
        label=f"Score vs. Cost ({split_name.capitalize()})"
    )

    # --- 3. Wire Up Event Handler for the Refresh Button ---
    # This connects the button specific to THIS tab.
    refresh_button.click(
        fn=load_display_data_for_split,
        inputs=[gr.State(split_name), global_dropdown],
        outputs=[df_output, scatter_plot_output, global_dropdown]
    )

    # --- 4. Return Components That Need to Be Accessed Globally ---
    # We need to return these so the GLOBAL dropdown can update them.
    return df_output, scatter_plot_output



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

    # --- Leaderboard Display Section ---
    gr.Markdown("---")
    gr.Markdown("## Leaderboard Results")

    with gr.Row():
        # Define the global dropdown. Its choices will be populated by the first tab.
        global_tag_dropdown = gr.Dropdown(
            choices=["Overall"], # Start with a default
            value="Overall",
            label="Filter by Tag/Capability",
        )

    # Use the helper function to create the tabs, avoiding duplicate code
    with gr.Tabs():
        with gr.Tab("Results: Validation"):
            # This creates all components and sets up the refresh button event
            val_df_output, val_scatter_plot_output = create_leaderboard_tab(
                split_name="validation",
                global_dropdown=global_tag_dropdown
            )

        with gr.Tab("Results: Test"):
            test_df_output, test_scatter_plot_output = create_leaderboard_tab(
                split_name="test",
                global_dropdown=global_tag_dropdown
            )

    # --- FINAL CRITICAL STEP: Add the missing event handler for the dropdown ---
    # This single event handler updates ALL relevant components in BOTH tabs
    # whenever the dropdown value changes.

    # We need a small wrapper because the dropdown doesn't know which split is active.
    # This wrapper will call the main loading function for both splits.
    def update_all_tabs(selected_tag: str):
        val_df, val_plot, val_tags = load_display_data_for_split("validation", selected_tag)
        test_df, test_plot, _ = load_display_data_for_split("test", selected_tag) # Tags are the same
        return val_df, val_plot, test_df, test_plot, val_tags # Return all updated components

    global_tag_dropdown.change(
        fn=update_all_tabs,
        inputs=[global_tag_dropdown],
        outputs=[
            val_df_output,
            val_scatter_plot_output,
            test_df_output,
            test_scatter_plot_output,
            global_tag_dropdown  # The last returned item updates the dropdown itself
        ]
    )

    with gr.Accordion("📙 Citation", open=False):
        gr.Textbox(value=CITATION_BUTTON_TEXT, label=CITATION_BUTTON_LABEL, elem_id="citation-button-main", interactive=False)


if __name__ == "__main__":
    demo.launch()