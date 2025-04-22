"""app.py: Gradio app for the AstaBench leaderboard.

Modeled after the GAIA huggingface leaderboard app.

"""

import json
import os
import re
import shutil
import tarfile
import tempfile
from datetime import datetime, timedelta, timezone
from email.utils import parseaddr
from zoneinfo import ZoneInfo

import gradio as gr
import numpy as np
import pandas as pd
import requests
from agenteval import (
    compute_summary_statistics,
    score_directory,
    upload_folder_to_hf,
    upload_summary_to_hf,
)
from agenteval.models import EvalResult
from apscheduler.schedulers.background import BackgroundScheduler
from datasets import Dataset, DatasetDict, VerificationMode, load_dataset
from datasets.data_files import EmptyDatasetError
from huggingface_hub import HfApi

from content import (
    CITATION_BUTTON_LABEL,
    CITATION_BUTTON_TEXT,
    INTRODUCTION_TEXT,
    SUBMISSION_TEXT,
    TITLE,
    format_error,
    format_log,
    format_warning,
    hf_uri_to_web_url,
    hyperlink,
)

# Should be False on spaces and True outside
LOCAL_DEBUG = not (os.environ.get("system") == "spaces")


CONFIG_NAME = "1.0.0-dev1"

IS_INTERNAL = os.environ.get("IS_INTERNAL", "false").lower() == "true"

OWNER = "allenai"
PROJECT_NAME = "asta-bench" + ("-internal" if IS_INTERNAL else "")
SUBMISSION_DATASET = f"{OWNER}/{PROJECT_NAME}-submissions"  # all raw and scored submissions (val and test)
SUBMISSION_DATASET_PUBLIC = f"{OWNER}/{PROJECT_NAME}-submissions-public"  # copy scored val submissions (public for transparency - not used for rendering leaderboard)
CONTACT_DATASET = f"{OWNER}/{PROJECT_NAME}-contact-info"
RESULTS_DATASET = f"{OWNER}/{PROJECT_NAME}-results"  # just the summary score statistics (val and test), to be displayed on the leaderboard
LEADERBOARD_PATH = f"{OWNER}/{PROJECT_NAME}-leaderboard"

if LOCAL_DEBUG:
    DATA_DIR = os.path.join(os.path.dirname(__file__), "data", CONFIG_NAME)
else:
    DATA_DIR = "/home/user/data/" + CONFIG_NAME
EXTRACTED_DATA_DIR = os.path.join(DATA_DIR, "extracted")

api = HfApi()


os.makedirs(EXTRACTED_DATA_DIR, exist_ok=True)


def try_load_dataset(*args, **kwargs) -> DatasetDict:
    try:
        return load_dataset(*args, **kwargs)
    except EmptyDatasetError:
        return DatasetDict()
    except ValueError:
        return DatasetDict()


def pretty_column_name(col: str) -> str:
    """Map any raw column name to its display name."""
    # text columns
    if col == "submit_time":
        return "Submission date"
    elif col == "agent_name":
        return "Agent"
    elif col == "agent_description":
        return "Agent description"
    elif col == "username":
        return "User/organization"
    elif col == "logs_url":
        return "Logs"
    # cost → $
    if col.endswith("/cost"):
        return "$"
    # stderr → CI
    elif col.endswith("/cost_stderr") or col.endswith("/score_stderr"):
        return "CI"
    # overall score
    elif col == "overall/score":
        return "Overall"
    # any other score → its category/task name
    elif col.endswith("/score"):
        return col.split("/")[1]
    # fallback to unchanged
    return col


def get_dataframe_from_results(eval_results: DatasetDict, split: str):
    local_df = eval_results.get(split)
    # return default if split is missing or contains no records
    if local_df is None or len(local_df) == 0:
        default_raw_cols = [
            "agent_name",
            "agent_description",
            "username",
            "submit_time",
        ]
        pretty_cols = [pretty_column_name(c) for c in default_raw_cols]
        return pd.DataFrame({col: ["No data"] for col in pretty_cols})

    # Use the first suite_config for all rows
    # because the suite_config should not change given a single CONFIG_NAME
    first_suite_config = None
    if len(local_df) > 0:
        first_suite_config = EvalResult.model_validate(local_df[0]).suite_config

    def extract_scores(eval_res: EvalResult) -> dict[str, float | None]:
        summary_stats = compute_summary_statistics(
            suite_config=first_suite_config,
            split=split,
            results=eval_res.results,
        )

        values: dict[str, float | None] = {}
        for key in summary_stats:
            if key == "overall":
                values["overall/score"] = summary_stats[key].score
                values["overall/cost"] = summary_stats[key].cost
            elif key.startswith("category/"):
                category = key.split("/")[1]
                values[f"category/{category}/score"] = summary_stats[key].score
                values[f"category/{category}/cost"] = summary_stats[key].cost
            elif key.startswith("task/"):
                task = key.split("/")[1]
                values[f"task/{task}/score"] = summary_stats[key].score
                values[f"task/{task}/score_stderr"] = summary_stats[key].score_stderr
                values[f"task/{task}/cost"] = summary_stats[key].cost
                values[f"task/{task}/cost_stderr"] = summary_stats[key].cost_stderr
        return values

    def format_row(row) -> dict[str, float | str | None]:
        eval_res = EvalResult.model_validate(row)
        sub = eval_res.submission
        sub.submit_time = sub.submit_time or datetime(1970, 1, 1, 0, 0, 0)
        data = {
            "submit_time": sub.submit_time.astimezone(ZoneInfo("US/Pacific")).strftime(
                "%Y-%m-%d"
            ),
            "agent_name": (
                hyperlink(sub.agent_url, sub.agent_name)
                if sub.agent_url
                else sub.agent_name
            ),
            "agent_description": sub.agent_description or "",
            "username": sub.username or "",
            **extract_scores(eval_res),
            "logs_url": (
                hyperlink(
                    hf_uri_to_web_url(
                        sub.logs_url if IS_INTERNAL else sub.logs_url_public
                    ),
                    "🔗",
                )
                if (sub.logs_url or sub.logs_url_public)
                else ""
            ),
        }
        return data

    local_df = local_df.map(format_row)

    df = pd.DataFrame(local_df)

    # Multiply score, cost, and stderr values by 100 and round to 1 decimal
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df[numeric_cols] = df[numeric_cols].multiply(100).round(1)

    # Build column order on raw names, then rename via pretty_column_name
    all_cols = df.columns.tolist()
    base = ["agent_name", "agent_description", "username"]
    overall = ["overall/score", "overall/cost"]
    cats = sorted({c.split("/")[1] for c in all_cols if c.startswith("category/")})
    tasks = sorted({c.split("/")[1] for c in all_cols if c.startswith("task/")})
    rest = ["submit_time", "logs_url"]
    column_order = (
        base
        + overall
        + [
            col
            for cat in cats
            for col in (f"category/{cat}/score", f"category/{cat}/cost")
        ]
        + [
            col
            for t in tasks
            for col in (
                f"task/{t}/score",
                f"task/{t}/score_stderr",
                f"task/{t}/cost",
                f"task/{t}/cost_stderr",
            )
        ]
        + rest
    )
    df = df.reindex(columns=[c for c in column_order if c in all_cols])
    # sort by overall score (descending)
    df = df.sort_values(by=["overall/score"], ascending=False)
    # apply all renames via pretty_column_name
    orig_cols = df.columns.tolist()
    df.columns = [pretty_column_name(col) for col in orig_cols]

    # blank out any null/NaN cells
    df = df.fillna("")

    return df


def load_and_format_dataframes():
    eval_results = try_load_dataset(
        RESULTS_DATASET,
        CONFIG_NAME,
        download_mode="force_redownload",
        verification_mode=VerificationMode.NO_CHECKS,
        trust_remote_code=True,
    )
    eval_dataframe_val = get_dataframe_from_results(
        eval_results=eval_results, split="validation"
    )
    eval_dataframe_test = get_dataframe_from_results(
        eval_results=eval_results, split="test"
    )
    return eval_results, eval_dataframe_val, eval_dataframe_test


# Display the results
eval_results, eval_dataframe_val, eval_dataframe_test = load_and_format_dataframes()


def restart_space():
    api.restart_space(repo_id=LEADERBOARD_PATH)


def sanitize_name(name: str) -> str:
    """Sanitize a string to be safe for filenames."""
    return re.sub(r"[^A-Za-z0-9_-]", "_", name)


def add_new_eval(
    val_or_test: str,
    agent_name: str | None,
    agent_description: str,
    agent_url: str,
    path_to_file: tempfile._TemporaryFileWrapper | None,
    username: str,
    mail: str,
    profile: gr.OAuthProfile,
):
    # default username if none provided
    if not username or username.strip() == "":
        username = profile.username

    if not agent_name:
        return format_warning("Please provide an agent name.")

    submission_time = datetime.now(timezone.utc)

    # Was the profile created less than 2 month ago?
    user_data = requests.get(
        f"https://huggingface.co/api/users/{profile.username}/overview"
    )
    creation_date = json.loads(user_data.content)["createdAt"]

    created_at = datetime.strptime(creation_date, "%Y-%m-%dT%H:%M:%S.%fZ").replace(
        tzinfo=timezone.utc
    )
    if submission_time - created_at < timedelta(days=60):
        return format_error("This account is not authorized to submit here.")

    contact_infos = try_load_dataset(
        CONTACT_DATASET,
        CONFIG_NAME,
        download_mode="force_redownload",
        verification_mode=VerificationMode.NO_CHECKS,
        trust_remote_code=True,
    )
    user_submission_dates = sorted(
        datetime.fromisoformat(row["submit_time"])
        for row in contact_infos.get(val_or_test, [])
        if row["username_auth"] == profile.username
    )
    if len(user_submission_dates) > 0 and abs(
        submission_time - user_submission_dates[-1]
    ) < timedelta(seconds=24 * 60 * 60):
        return format_error(
            "You already submitted once in the last 24h; please try again later."
        )

    is_validation = val_or_test == "validation"

    # Very basic email parsing
    _, parsed_mail = parseaddr(mail)
    if "@" not in parsed_mail:
        return format_warning("Please provide a valid email adress.")

    # Check duplicate submissions by inspecting the nested "submission" dicts
    if val_or_test in eval_results and len(eval_results[val_or_test]) > 0:
        existing = eval_results[val_or_test]
        subs = existing.to_dict().get("submission", [])
        names = {item.get("agent_name", "").lower() for item in subs}
        users = {item.get("username", "").lower() for item in subs}
        if agent_name.lower() in names and username.lower() in users:
            return format_warning("This agent has been already submitted.")

    if path_to_file is None:
        return format_warning("Please attach a file.")

    # sanitize username and agent_name for filesystem
    safe_username = sanitize_name(username)
    safe_agent_name = sanitize_name(agent_name)

    extracted_dir = os.path.join(
        EXTRACTED_DATA_DIR, f"{safe_username}_{safe_agent_name}"
    )

    if LOCAL_DEBUG:
        print("mock extracted file", flush=True)
    else:
        try:
            # 1) remove old extraction if present
            if os.path.exists(extracted_dir):
                shutil.rmtree(extracted_dir)
            # ensure directory exists
            os.makedirs(extracted_dir, exist_ok=True)

            # 2) extract all members
            with tarfile.open(path_to_file.name, "r:gz") as tar:
                tar.extractall(extracted_dir)

            # 3) flatten nested directories
            for root, _, files in os.walk(extracted_dir):
                for file in files:
                    src = os.path.join(root, file)
                    dst = os.path.join(extracted_dir, file)
                    if src != dst:
                        os.replace(src, dst)

            # 4) remove hidden files/dirs, then empty dirs
            for root, dirs, files in os.walk(extracted_dir, topdown=False):
                # delete all hidden files
                for name in files:
                    if name.startswith("."):
                        os.remove(os.path.join(root, name))
                # delete hidden dirs (and contents), then try to remove empty dirs
                for name in dirs:
                    full_path = os.path.join(root, name)
                    if name.startswith("."):
                        shutil.rmtree(full_path)
                    else:
                        try:
                            os.rmdir(full_path)
                        except OSError:
                            pass

        except Exception as e:
            return format_error(
                f"Error while extracting the file: {e}. Be sure to upload a valid .tar.gz file."
            )

    submission_name = (
        f"{safe_username}_{safe_agent_name}_{submission_time.strftime('%Y-%m-%d')}"
    )

    # SAVE UNSCORED SUBMISSION
    if LOCAL_DEBUG:
        print("mock uploaded submission", flush=True)
    else:
        upload_folder_to_hf(
            api=api,
            folder_path=extracted_dir,
            repo_id=SUBMISSION_DATASET,
            config_name=CONFIG_NAME,
            split=val_or_test,
            submission_name=submission_name,
        )

    # SAVE CONTACT
    contact_info = {
        "agent_name": agent_name,
        "agent_description": agent_description,
        "url": agent_url,
        "username": username,
        "username_auth": profile.username,
        "mail": mail,
        "submit_time": submission_time.isoformat(),
    }
    # add or init contact dataset for this split
    if val_or_test in contact_infos:
        contact_infos[val_or_test] = contact_infos[val_or_test].add_item(contact_info)
    else:
        contact_infos[val_or_test] = Dataset.from_list([contact_info])
    if LOCAL_DEBUG:
        print("mock uploaded contact info", flush=True)
    else:
        contact_infos.push_to_hub(CONTACT_DATASET, config_name=CONFIG_NAME)

    try:
        # NOTE: Trusting uploaded results. This just aggregates the scores.
        eval_result = score_directory(
            log_dir=extracted_dir,
        )
        if eval_result.suite_config.version != CONFIG_NAME:
            return format_error(
                f"Error: the version of the uploaded submission ({eval_result.suite_config.version}) does not match the current version ({CONFIG_NAME})."
            )
        if eval_result.split != val_or_test:
            return format_error(
                f"Error: the split of the uploaded submission ({eval_result.split}) does not match the selected split ({val_or_test})."
            )
        eval_result.save_json(
            os.path.join(extracted_dir, "agenteval.json"),
        )
    except Exception as e:
        return format_error(
            f"Error while scoring the submission: {e}. Be sure to upload a valid submission."
        )

    # # SAVE SCORED SUBMISSION
    if LOCAL_DEBUG:
        print("mock uploaded scored submission")
    else:
        logs_url_private, _ = upload_folder_to_hf(
            api=api,
            folder_path=extracted_dir,
            repo_id=SUBMISSION_DATASET,
            config_name=CONFIG_NAME,
            split=val_or_test,
            submission_name=f"{submission_name}_scored",
        )

        # Validation submissions are public for public leaderboard
        if is_validation and not IS_INTERNAL:
            logs_url_public, _ = upload_folder_to_hf(
                api=api,
                folder_path=extracted_dir,
                repo_id=SUBMISSION_DATASET_PUBLIC,
                config_name=CONFIG_NAME,
                split=val_or_test,
                submission_name=f"{submission_name}_scored",
            )
        else:
            logs_url_public = None

    eval_result.submission.agent_name = agent_name
    eval_result.submission.agent_description = agent_description
    eval_result.submission.agent_url = agent_url
    eval_result.submission.username = username
    eval_result.submission.submit_time = submission_time
    eval_result.submission.logs_url = logs_url_private
    eval_result.submission.logs_url_public = logs_url_public

    if LOCAL_DEBUG:
        print("mock uploaded results to lb")
    else:
        upload_summary_to_hf(
            api=api,
            eval_result=eval_result,
            repo_id=RESULTS_DATASET,
            config_name=CONFIG_NAME,
            split=val_or_test,
            submission_name=f"{submission_name}_scored",
        )

    return format_log(
        f"Agent {agent_name} submitted by {username} successfully.\nPlease wait a few hours and refresh the leaderboard to see your score displayed."
    )


def refresh():
    _, eval_dataframe_val, eval_dataframe_test = load_and_format_dataframes()
    return eval_dataframe_val, eval_dataframe_test


# Determine column types dynamically based on dataframe columns
def compute_column_types(df):
    col_types = []
    for col in df.columns:
        if col == "Agent":
            col_types.append("markdown")
        elif col in ["Agent description", "User/organization", "Submission date"]:
            col_types.append("str")
        elif col == "Logs":
            col_types.append("markdown")
        else:
            col_types.append("number")
    return col_types


test_col_types = compute_column_types(eval_dataframe_test)
val_col_types = compute_column_types(eval_dataframe_val)

demo = gr.Blocks()
with demo:
    gr.HTML(TITLE)
    gr.Markdown(INTRODUCTION_TEXT, elem_classes="markdown-text")

    with gr.Row():
        with gr.Accordion("📙 Citation", open=False):
            citation_button = gr.Textbox(
                value=CITATION_BUTTON_TEXT,
                label=CITATION_BUTTON_LABEL,
                elem_id="citation-button",
            )  # .style(show_copy_button=True)

    leaderboard_table_test = gr.Dataframe(
        value=eval_dataframe_test,
        headers=list(eval_dataframe_test.columns),
        datatype=test_col_types,
        interactive=False,
        column_widths=["20%"],
        render=False,
    )

    leaderboard_table_val = gr.Dataframe(
        value=eval_dataframe_val,
        headers=list(eval_dataframe_val.columns),
        datatype=val_col_types,
        interactive=False,
        column_widths=["20%"],
        render=False,
    )

    # Build tab layout list based on desired order
    tabs = [
        ("Results: Test", leaderboard_table_test),
        ("Results: Validation", leaderboard_table_val),
    ]

    if IS_INTERNAL:
        tabs = [tabs[1], tabs[0]]  # Validation first for internal users

    # Render the tabs in desired order
    for label, component in tabs:
        with gr.Tab(label):
            component.render()

    refresh_button = gr.Button("Refresh")
    refresh_button.click(
        refresh,
        inputs=[],
        outputs=[
            leaderboard_table_val,
            leaderboard_table_test,
        ],
    )
    with gr.Accordion("Submit a new agent for evaluation"):
        with gr.Row():
            gr.Markdown(SUBMISSION_TEXT, elem_classes="markdown-text")
        with gr.Row():
            with gr.Column():
                level_of_test = gr.Radio(
                    ["validation", "test"], value="validation", label="Split"
                )
                agent_name_textbox = gr.Textbox(label="Agent name")
                agent_description_textbox = gr.Textbox(label="Agent description")
                agent_url_textbox = gr.Textbox(label="Url to agent information")
            with gr.Column():
                username = gr.Textbox(
                    label="Organization or user name (defaults to your HF username)",
                    placeholder="Leave blank to use your HF username",
                )
                mail = gr.Textbox(
                    label="Contact email (will be stored privately, & used if there is an issue with your submission)"
                )
                file_output = gr.File()

        with gr.Row():
            gr.LoginButton()
            submit_button = gr.Button("Submit Eval")
        submission_result = gr.Markdown()
        submit_button.click(
            add_new_eval,
            [
                level_of_test,
                agent_name_textbox,
                agent_description_textbox,
                agent_url_textbox,
                file_output,
                username,
                mail,
            ],
            submission_result,
        )

scheduler = BackgroundScheduler()
scheduler.add_job(restart_space, "interval", seconds=3600)
scheduler.start()
if LOCAL_DEBUG:
    demo.launch(debug=True)
else:
    demo.launch(server_name="0.0.0.0", server_port=7860, debug=True)
