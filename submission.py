import logging
import sys

import matplotlib
from agenteval.cli import SUBMISSION_METADATA_FILENAME
from agenteval.models import SubmissionMetadata

matplotlib.use('Agg')

import os
import shutil
import tarfile
import tempfile
from datetime import datetime, timedelta, timezone
from email.utils import parseaddr
from pathlib import Path

import gradio as gr
import requests
from agenteval import (
    process_eval_logs,
    upload_folder_to_hf,
)
from agenteval.leaderboard.models import LeaderboardSubmission
from agenteval.leaderboard.upload import sanitize_path_component
from datasets import Dataset, DatasetDict, VerificationMode, load_dataset
from datasets.data_files import EmptyDatasetError
from huggingface_hub import HfApi

from config import (
    CONFIG_NAME,
    CONTACT_DATASET,
    EXTRACTED_DATA_DIR,
    IS_INTERNAL,
    LOCAL_DEBUG,
    RESULTS_DATASET,
    SUBMISSION_DATASET,
)
from content import (
    CITATION_BUTTON_LABEL,
    CITATION_BUTTON_TEXT,
    format_error,
    format_log,
    format_warning,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

api = HfApi()
MAX_UPLOAD_BYTES = 100 * 1024**2
AGENTEVAL_MANIFEST_NAME = "agenteval.json"
os.makedirs(EXTRACTED_DATA_DIR, exist_ok=True)

# --- Submission Logic (largely unchanged from original, ensure LeaderboardSubmission and other deps are fine) ---
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
        email: str,
        profile: gr.OAuthProfile,
):
    if not agent_name:
        return format_warning("Please provide an agent name.")

    def submission_error(msg: str) -> str:
        logger.debug(f"agent {agent_name}: {msg}")
        return format_error(msg)

    if path_to_file is None:
        return format_warning("Please attach a .tar.gz file.")

    logger.info(f"agent {agent_name}: Checking submission")

    # Load current eval_results for submission checks
    # This is a bit redundant if display part reloads it, but submission needs its own consistent view
    current_eval_results_for_submission = try_load_dataset_submission(
        RESULTS_DATASET,
        CONFIG_NAME,
        download_mode="force_redownload", # Or a less aggressive mode
        verification_mode=VerificationMode.NO_CHECKS,
    )

    submission_time = datetime.now(timezone.utc)
    if not username or username.strip() == "":
        username = profile.username # Default to HF username

    logger.debug(f"agent {agent_name}: User account age check {profile.username}")
    try:
        user_data_resp = requests.get(f"https://huggingface.co/api/users/{profile.username}/overview")
        user_data_resp.raise_for_status()
        creation_date_str = user_data_resp.json()["createdAt"]
        created_at = datetime.strptime(creation_date_str, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone.utc)
        if submission_time - created_at < timedelta(days=60):
            return format_error("This account is not authorized to submit here (account too new).")
    except Exception as e:
        logger.warning(f"Error checking user account age: {e}")
        return submission_error("Could not verify account age. Please try again later.")

    logger.debug(f"agent {agent_name}: Submission frequency check {profile.username}")
    contact_infos = try_load_dataset_submission(
        CONTACT_DATASET, CONFIG_NAME, download_mode="force_redownload",
        verification_mode=VerificationMode.NO_CHECKS, trust_remote_code=True
    )
    user_submission_dates = sorted(
        datetime.fromisoformat(row["submit_time"])
        for row in contact_infos.get(val_or_test, []) if row["username_auth"] == profile.username
    )
    if user_submission_dates and (submission_time - user_submission_dates[-1] < timedelta(days=1)):
        logger.info(f"agent {agent_name}: Denied submission because user {username} submitted recently")
        return format_error("You already submitted once in the last 24h for this split; please try again later.")

    logger.debug(f"agent {agent_name}: Email validation {email}")
    _, parsed_mail = parseaddr(email)
    if "@" not in parsed_mail:
        return format_warning("Please provide a valid email address.")

    logger.debug(f"agent {agent_name}: Duplicate submission check")
    if val_or_test in current_eval_results_for_submission and len(current_eval_results_for_submission[val_or_test]) > 0:
        existing_submissions = current_eval_results_for_submission[val_or_test].to_dict().get("submission", [])
        for sub_item in existing_submissions:
            if (sub_item.get("agent_name", "").lower() == agent_name.lower() and
                    sub_item.get("username", "").lower() == username.lower()):
                return format_warning("This agent name by this user has already been submitted to this split.")

    safe_username = sanitize_path_component(username)
    safe_agent_name = sanitize_path_component(agent_name)
    extracted_dir = os.path.join(EXTRACTED_DATA_DIR, f"{safe_username}_{safe_agent_name}")

    logger.debug(f"agent {agent_name}: File extraction to {extracted_dir}")
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
                return submission_error("Submission tarball is empty or contains no valid files.")
    except Exception as e:
        return submission_error(f"Error extracting file: {e}. Ensure it's a valid .tar.gz.")

    submission_name = f"{safe_username}_{safe_agent_name}_{submission_time.strftime('%Y-%m-%d_%H-%M-%S')}"

    logger.debug(f"agent {agent_name}: Generate submission.json")
    subm_meta = SubmissionMetadata(
        agent_name=agent_name,
        agent_description=agent_description,
        agent_url=agent_url,
        openness=openness,
        tool_usage=degree_of_control,
        username=username,
        submit_time=submission_time,
    )
    with open(os.path.join(extracted_dir, SUBMISSION_METADATA_FILENAME), "w", encoding="utf-8") as fp:
        fp.write(subm_meta.model_dump_json(indent=2))

    logger.info(f"agent {agent_name}: Upload raw (unscored) submission files")
    try:
        checked_upload_folder(api, extracted_dir, SUBMISSION_DATASET, CONFIG_NAME, val_or_test, submission_name)
    except ValueError as e:
        return submission_error(str(e))
    except Exception as e:
        return submission_error(f"Failed to upload raw submission: {e}")

    logger.info(f"agent {agent_name}: Save contact information")
    contact_info = subm_meta.model_dump()
    contact_info["username_auth"] = profile.username
    contact_info["email"] = email
    logger.debug(f"agent {agent_name}: Contact info: {contact_info}")
    if val_or_test in contact_infos:
        contact_infos[val_or_test] = contact_infos[val_or_test].add_item(contact_info)
    else:
        contact_infos[val_or_test] = Dataset.from_list([contact_info])

    try:
        contact_infos.push_to_hub(CONTACT_DATASET, config_name=CONFIG_NAME)
    except Exception as e:
        return submission_error(f"Submission recorded, but contact info failed to save: {e}")

    msg = f"Agent '{agent_name}' submitted successfully by '{username}' to '{val_or_test}' split. "
    logger.info(f"agent {agent_name}: {msg}")
    return format_log(msg)

def _deprecated_scoring_logic():
    # No longer triggered on eval submission. Kept for quick reference for a little while (2025). TODO delete this.

    # 3. Process and score the submission
    eval_result_obj = None # Define to avoid NameError
    try:
        json_path = Path(extracted_dir) / AGENTEVAL_MANIFEST_NAME
        if not json_path.exists():
            return format_error(f"Missing manifest {AGENTEVAL_MANIFEST_NAME} in submission.")

        eval_result_obj = LeaderboardSubmission.model_validate_json(json_path.read_text(encoding="utf-8"))
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


    # Update LeaderboardSubmission with submission details
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

    return format_log(
        f"Agent '{agent_name}' submitted successfully by '{username}' to '{val_or_test}' split. "
        "Please refresh the leaderboard in a few moments. It may take some time for changes to propagate."
    )

openness_label_html = """
<div class="form-label-with-tooltip">
    Openness of Agent
    <span class="tooltip-icon" data-tooltip="• Closed: No API or code available
        • API Available: API available, but no code
        • Open Source: Code available, but no weights
        • Open Source + Open Weights: Code and weights available"
    >
        ⓘ
    </span>
</div>
"""

agent_tooling_label_html = """
<div class="form-label-with-tooltip">
    Agent Tooling
    <span class="tooltip-icon" data-tooltip="• Standard: Only uses tools explicitly provided in state.tools
        • Equivalent: Uses custom tools with identical or more restricted capabilities
        • Fully Custom: Uses tools beyond constraints of Standard or Equivalent"
    >
        ⓘ
    </span>
</div>
"""

heading_html = """
<h2>🚀 Submit an agent for evaluation</h2>
<p>Submit your agent to AstaBench for evaluation on real-world scientific tasks. Once submitted, your run will be reviewed by our team. If there are any issues, we’ll reach out within 5–7 business days. We’re working toward full automation, but in the meantime, human review helps ensure quality and trust.</p>
<h3>How to run an evaluation</h3>
<p>Please follow the steps in our <a href="https://github.com/allenai/asta-bench" target="_blank">README</a>. You’ll upload your run file at the end of this form.</p>
"""

# --- Submission Accordion ---
def build_page():
    with gr.Row():
        with gr.Column():
            gr.HTML(heading_html)
        with gr.Column():
            pass # Keeps this row's content on the left side of the page.
    with gr.Row():
        gr.LoginButton()
    with gr.Row():
        with gr.Column():
            level_of_test_radio = gr.Radio(["validation", "test"], value="validation", label="Split")
            agent_name_tb = gr.Textbox(label="Agent Name")
            agent_desc_tb = gr.Textbox(label="Agent Description")
            agent_url_tb = gr.Textbox(label="URL to Agent Information")
            with gr.Group(elem_id="custom-form-group"):
                gr.HTML(value=openness_label_html, elem_id="openness-label-html")
                openness_radio = gr.Radio(["Open Source","Open Source Open Weights", "API Available", "Closed"], value=None, label="")
                gr.HTML(value=agent_tooling_label_html, elem_id="agent-tooling-label-html")
                degree_of_control_radio = gr.Radio(["Standard","Equivalent", "Fully Custom"], value=None, label="")
        with gr.Column():
            username_tb = gr.Textbox(label="Organization or User Name (Defaults to HF username)")
            mail_tb = gr.Textbox(label="Contact Email (Private, for submission issues)")
            file_upload_comp = gr.File(
                label="Submission File (.tar.gz ...)", # Shortened for brevity
                file_types=[".gz", ".tar.gz"]
            )
    with gr.Row():
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
    with gr.Accordion("📙 Citation", open=False):
        gr.Textbox(value=CITATION_BUTTON_TEXT, label=CITATION_BUTTON_LABEL, elem_id="citation-button-main", interactive=False)
