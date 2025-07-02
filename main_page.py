import matplotlib
matplotlib.use('Agg')

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
from datasets import Dataset, DatasetDict, VerificationMode, load_dataset # load_dataset used by LV
from datasets.data_files import EmptyDatasetError
from huggingface_hub import HfApi

from ui_components import create_leaderboard_display, get_full_leaderboard_data

from content import (
    CITATION_BUTTON_LABEL,
    CITATION_BUTTON_TEXT,
    SUBMISSION_TEXT,
    INTRO_PARAGRAPH,
    format_error,
    format_log,
    format_warning,
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
        openness: str | None,
        degree_of_control: str | None,
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
    eval_result_obj.submission.openness = openness
    eval_result_obj.submission.degree_of_control = degree_of_control
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

with gr.Blocks(fill_width=True) as demo:
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
                openness_radio = gr.Radio(["Open Source","Open Source Open Weights" "API", "UI Only"], value=None, label="Openness of Agent")
                degree_of_control_radio = gr.Radio(["Standard", "Custom"], value=None, label="Degree of Control")
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
                openness_radio,
                degree_of_control_radio,
                file_upload_comp,
                username_tb,
                mail_tb
            ],
            submission_result,
        )

    # --- Leaderboard Display Section ---
    gr.Markdown("---")
    CATEGORY_NAME = "Overall"
    gr.Markdown(f"## {CATEGORY_NAME} Leaderboard Results")

    with gr.Tabs() as tabs:
        with gr.Tab("Results: Validation"):
            # 1. Load all necessary data for the "validation" split ONCE.
            validation_df, validation_tag_map = get_full_leaderboard_data("validation")

            # Check if data was loaded successfully before trying to display it
            if not validation_df.empty:
                # 2. Render the display by calling the factory with the loaded data.
                create_leaderboard_display(
                    full_df=validation_df,
                    tag_map=validation_tag_map,
                    category_name=CATEGORY_NAME, # Use our constant
                    split_name="validation"
                )
            else:
                gr.Markdown("No data available for validation split.")

        with gr.Tab("Results: Test"):
            test_df, test_tag_map = get_full_leaderboard_data("test")
            if not test_df.empty:
                create_leaderboard_display(
                    full_df=test_df,
                    tag_map=test_tag_map,
                    category_name=CATEGORY_NAME, # Use our constant
                    split_name="test"
                )
            else:
                gr.Markdown("No data available for test split.")

    with gr.Accordion("📙 Citation", open=False):
        gr.Textbox(value=CITATION_BUTTON_TEXT, label=CITATION_BUTTON_LABEL, elem_id="citation-button-main", interactive=False)


if __name__ == "__main__":
    demo.launch()