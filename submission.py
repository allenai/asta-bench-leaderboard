import logging

import matplotlib
from agenteval.cli import SUBMISSION_METADATA_FILENAME, EVAL_CONFIG_FILENAME
from agenteval.models import SubmissionMetadata
from datasets.exceptions import DataFilesNotFoundError
from gradio_modal import Modal

matplotlib.use('Agg')

import os
import shutil
import tarfile
import tempfile
from datetime import datetime, timedelta, timezone
from email.utils import parseaddr

import gradio as gr
import requests
from agenteval import upload_folder_to_hf
from agenteval.leaderboard.upload import sanitize_path_component
from datasets import Dataset, DatasetDict, VerificationMode, load_dataset
from datasets.data_files import EmptyDatasetError
from huggingface_hub import HfApi

from config import (
    CONFIG_NAME,
    CONTACT_DATASET,
    EXTRACTED_DATA_DIR,
    RESULTS_DATASET,
    SUBMISSION_DATASET,
)
from content import (
    CITATION_BUTTON_LABEL,
    CITATION_BUTTON_TEXT,
    SUBMISSION_CONFIRMATION,
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
    except DataFilesNotFoundError:
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

def show_loading_spinner():
    return gr.update(visible=True)

def add_new_eval(
        val_or_test: str,
        agent_name: str | None,
        agent_description: str,
        agent_url: str,
        openness: str | None,
        degree_of_control: str | None,
        path_to_file: tempfile._TemporaryFileWrapper | None,
        username: str,
        role: str,
        email: str,
        email_opt_in: bool,
        profile: gr.OAuthProfile,
):
    if not agent_name:
        return (
            format_warning("Please provide an agent name."),  # error_message
            gr.update(visible=True),                            # error_modal
            gr.update(visible=False),                           # success_modal
            gr.update(visible=False)                            # loading_modal
        )

    if path_to_file is None:
        return (
            format_warning("Please attach a .tar.gz file."),  # error_message
            gr.update(visible=True),                            # error_modal
            gr.update(visible=False),                           # success_modal
            gr.update(visible=False)                            # loading_modal
        )

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
        # Account age check disabled for launch.
        # https://github.com/allenai/astabench-issues/issues/419
        # if _is_hf_acct_too_new(submission_time, profile.username):
        #     return (
        #         format_error("This account is not authorized to submit here (account too new)."),  # error_message
        #         gr.update(visible=True),                            # error_modal
        #         gr.update(visible=False),                           # success_modal
        #         gr.update(visible=False)                            # loading_modal
        #     )
        pass
    except Exception as e:
        logger.warning(f"Error checking user account age: {e}")
        return (
            format_error("Could not verify account age. Please try again later."),  # error_message
            gr.update(visible=True),                            # error_modal
            gr.update(visible=False),                           # success_modal
            gr.update(visible=False)                            # loading_modal
        )

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
        return (
            format_error("You already submitted once in the last 24h for this split; please try again later."),  # error_message
            gr.update(visible=True),                            # error_modal
            gr.update(visible=False),                           # success_modal
            gr.update(visible=False)                            # loading_modal
        )

    logger.debug(f"agent {agent_name}: Email validation {email}")
    _, parsed_mail = parseaddr(email)
    if "@" not in parsed_mail:
        return (
            format_warning("Please provide a valid email address."),  # error_message
            gr.update(visible=True),                            # error_modal
            gr.update(visible=False),                           # success_modal
            gr.update(visible=False)                            # loading_modal
        )

    logger.debug(f"agent {agent_name}: Duplicate submission check")
    if val_or_test in current_eval_results_for_submission and len(current_eval_results_for_submission[val_or_test]) > 0:
        existing_submissions = current_eval_results_for_submission[val_or_test].to_dict().get("submission", [])
        for sub_item in existing_submissions:
            if (sub_item.get("agent_name", "").lower() == agent_name.lower() and
                    sub_item.get("username", "").lower() == username.lower()):
                return (
                    format_warning("This agent name by this user has already been submitted to this split."),  # error_message
                    gr.update(visible=True),                            # error_modal
                    gr.update(visible=False),                           # success_modal
                    gr.update(visible=False)                            # loading_modal
                )

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
                # Keep only specific files as part of submissions.
                if not fname.endswith(".eval") or fname == EVAL_CONFIG_FILENAME: continue
                fobj = tar.extractfile(member)
                if not fobj: continue
                with open(os.path.join(extracted_dir, fname), "wb") as out:
                    out.write(fobj.read())
                members_extracted +=1
            if members_extracted == 0:
                return (
                    format_error("Submission tarball is empty or contains no valid files."),  # error_message
                    gr.update(visible=True),                            # error_modal
                    gr.update(visible=False),                           # success_modal
                    gr.update(visible=False)                            # loading_modal
                )
    except Exception as e:
        return (
            format_error(f"Error extracting file: {e}. Ensure it's a valid .tar.gz."),  # error_message
            gr.update(visible=True),                            # error_modal
            gr.update(visible=False),                           # success_modal
            gr.update(visible=False)                            # loading_modal
        )

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
        return (
            format_error(str(e)),                               # error_message
            gr.update(visible=True),                            # error_modal
            gr.update(visible=False),                           # success_modal
            gr.update(visible=False)                            # loading_modal
        )
    except Exception as e:
        return (
            format_error(f"Failed to upload raw submission: {e}"),  # error_message
            gr.update(visible=True),                            # error_modal
            gr.update(visible=False),                           # success_modal
            gr.update(visible=False)                            # loading_modal
        )

    logger.info(f"agent {agent_name}: Save contact information")
    contact_info = subm_meta.model_dump()
    contact_info["submit_time"] = submission_time.isoformat()
    contact_info["username_auth"] = profile.username
    contact_info["email"] = email
    contact_info["email_opt_in"] = email_opt_in
    contact_info["role"] = role

    logger.debug(f"agent {agent_name}: Contact info: {contact_info}")
    if val_or_test in contact_infos:
        contact_infos[val_or_test] = contact_infos[val_or_test].add_item(contact_info)
    else:
        contact_infos[val_or_test] = Dataset.from_list([contact_info])

    try:
        contact_infos.push_to_hub(CONTACT_DATASET, config_name=CONFIG_NAME)
    except Exception as e:
        return (
            format_error(f"Submission recorded, but contact info failed to save: {e}"),  # error_message
            gr.update(visible=True),                            # error_modal
            gr.update(visible=False),                           # success_modal
            gr.update(visible=False)                            # loading_modal
        )

    logger.info(f"Agent '{agent_name}' submitted successfully by '{username}' to '{val_or_test}' split.")
    return (
        "",                                                 # error_message
        gr.update(visible=False),                           # error_modal
        gr.update(visible=True),                            # success_modal
        gr.update(visible=False)                            # loading_modal
    )


def _is_hf_acct_too_new(submission_time: datetime, username: str):
    user_data_resp = requests.get(f"https://huggingface.co/api/users/{username}/overview")
    user_data_resp.raise_for_status()
    creation_date_str = user_data_resp.json()["createdAt"]
    created_at = datetime.strptime(creation_date_str, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone.utc)
    return submission_time - created_at < timedelta(days=60)


openness_label_html = """
<div class="form-label-with-tooltip">
    Agent Openness
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
    with gr.Column(elem_id="submission-page-container"):
        gr.HTML(heading_html)
        gr.LoginButton()
        with gr.Group(elem_classes="custom-form-group"):
            gr.HTML(value="""<h2>Submitter Information</h2>""", elem_id="submitter-info-label-html")
            gr.HTML(value="""<h3>Username</h3>""", elem_classes="form-label")
            username_tb = gr.Textbox(label="This will show on the leaderboard. By default, we’ll use your Hugging Face username; but you can enter your organization name instead (e.g., university, company, or lab).")
            gr.HTML(value="""<h3>Role</h3>""", elem_classes="form-label")
            role = gr.Dropdown(label="Please select the role that most closely matches your current position. Helps us improve AstaBench for different user types. Not displayed on the leaderboard.",
                interactive=True,
                choices=[
                    "Undergraduate Student",
                    "Masters Student",
                    "PhD Student",
                    "Postdoctoral Researcher",
                    "Academic Faculty (e.g., Professor, Lecturer)",
                    "Industry Researcher (e.g., Research Scientist, Applied Scientist)",
                    "Engineer or Developer (e.g., Software or ML Engineer)",
                    "Data Scientist or Analyst",
                    "Product or Program Manager",
                    "Startup Founder or Independent Researcher",
                    "Other"
            ])
            gr.HTML(value="""<h3>Contact email</h3>""", elem_classes="form-label")
            mail_tb = gr.Textbox(label="We'll only use your email to communicate about your submission.")
            mail_opt_in = gr.Checkbox(label="I’m open to being contacted by email for user research studies or feedback opportunities.")
        with gr.Group(elem_classes="custom-form-group"):
            gr.HTML(value="""<h2>Agent Information</h2>""", elem_id="agent-info-label-html")
            gr.HTML(value="""<h3>Split</h3>""", elem_classes="form-label")
            level_of_test_radio = gr.Radio(choices=[
                ("Test set", "test"),
                ("Validation set", "validation"),
            ], elem_classes="form-label-fieldset", value="validation", label="The Test Set is used for final leaderboard rankings. The Validation Set is for development and iteration. Choose based on your evaluation goal.")
            gr.HTML(value="""<h3>Agent name</h3>""", elem_classes="form-label")
            agent_name_tb = gr.Textbox(label="This is how your agent will appear on the leaderboard. Use a clear, descriptive name (e.g., Asta Scholar QA, Perplexity Deep Research). Omit model names (e.g. GPT-4, Mistral) as they’ll be shown automatically based on your logs.")
            gr.HTML(value="""<h3>Agent description</h3>""", elem_classes="form-label")
            agent_desc_tb = gr.Textbox(label="Briefly describe your agent’s approach, core strategies, or what makes it distinct. This description may appear on the leaderboard.")
            gr.HTML(value="""<h3>URL</h3>""", elem_classes="form-label")
            agent_url_tb = gr.Textbox(label="Link to more information about your agent (e.g. GitHub repo, blog post, or website). This optional link may be shown on the leaderboard to let others explore your agent in more depth.")
            gr.HTML(value="""<h3>Agent openness</h3>""", elem_classes="form-label")
            openness_radio = gr.Radio(["Open Source","Open Source Open Weights", "API Available", "Closed"], elem_classes="form-label-fieldset", value=None, label="This affects how your submission is categorized on the leaderboard. Choose based on the availability of your code, model weights, or APIs.")
            gr.HTML(value="""<h3>Agent tooling</h3>""", elem_classes="form-label")
            degree_of_control_radio = gr.Radio(["Standard","Equivalent", "Fully Custom"], elem_classes="form-label-fieldset",value=None, label="Choose based on the tools and the execution environment your agent used during evaluation.")
            gr.HTML(value="""<h3>Submission file</h3>""", elem_classes="form-label")
            gr.HTML("<div id='submission-file-label'>Upload your run file, which is an archive prepared following the instructions in the <a href='https://github.com/allenai/asta-bench?tab=readme-ov-file#submitting-to-the-leaderboard' target='_blank'>README</a> (“Submitting to the Leaderboard”).</div>")
            file_upload_comp = gr.File(
                show_label=False,
                file_types=[".gz", ".tar.gz"]
            )
            submit_eval_button = gr.Button("Submit Evaluation", elem_id="submission-button")
    # Modals for loading spinner, success and error messages
    with Modal(visible=False, elem_id="submission-modal") as loading_modal:
        with gr.Column(elem_id="submission-modal-content"):
            gr.HTML('<div class="spinner-container"><div class="spinner"></div><p>Processing your submission...</p></div>')
    
    with Modal(visible=False, elem_id="submission-modal") as error_modal:
        with gr.Column(elem_id="submission-modal-content"):
            gr.Markdown("## ⚠️ Error")
            error_message = gr.Markdown()

    with Modal(visible=False, elem_id="submission-modal") as success_modal:
        with gr.Column(elem_id="submission-modal-content"):
            gr.Markdown(SUBMISSION_CONFIRMATION)

    submit_eval_button.click(
        show_loading_spinner,
        None,
        [loading_modal],
    ).then(
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
            role,
            mail_tb,
            mail_opt_in
        ],
        [error_message, error_modal, success_modal, loading_modal],
    )
    # hiding this for now till we have the real paper data
    # with gr.Accordion("📙 Citation", open=False):
    #     gr.Textbox(value=CITATION_BUTTON_TEXT, label=CITATION_BUTTON_LABEL, elem_id="citation-button-main", interactive=False)
