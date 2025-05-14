"""app.py: Gradio app for the AstaBench leaderboard.

Modeled after the GAIA huggingface leaderboard app.

"""

import json
import os
import shutil
import tarfile
import tempfile
from datetime import datetime, timedelta, timezone
from email.utils import parseaddr
from pathlib import Path
from zoneinfo import ZoneInfo
from dotenv import load_dotenv

import gradio as gr
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
from agenteval import (
    compute_summary_statistics,
    process_eval_logs,
    upload_folder_to_hf,
    upload_summary_to_hf,
)
from agenteval.models import EvalResult
from agenteval.upload import sanitize_path_component
from apscheduler.schedulers.background import BackgroundScheduler
from datasets import Dataset, DatasetDict, VerificationMode, load_dataset
from datasets.data_files import EmptyDatasetError
from huggingface_hub import HfApi

import content
from content import *
load_dotenv()
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

# max upload size of 100MB
MAX_UPLOAD_BYTES = 100 * 1024**2

AGENTEVAL_MANIFEST_NAME = "agenteval.json"


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
    # any other score → its tag/task name
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
        print(default_raw_cols)
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
            elif key.startswith("tag/"):
                tag = key.split("/")[1]
                values[f"tag/{tag}/score"] = summary_stats[key].score
                values[f"tag/{tag}/cost"] = summary_stats[key].cost
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
    tags = sorted({c.split("/")[1] for c in all_cols if c.startswith("tag/")})
    tasks = sorted({c.split("/")[1] for c in all_cols if c.startswith("task/")})
    rest = ["submit_time", "logs_url"]
    column_order = (
        base
        + overall
        + [col for tag in tags for col in (f"tag/{tag}/score", f"tag/{tag}/cost")]
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


def checked_upload_folder(
    api,
    folder_path: str,
    repo_id: str,
    config_name: str,
    split: str,
    submission_name: str,
) -> str:
    """Upload with inline size check; raises ValueError if too large."""
    total = 0
    for root, _, files in os.walk(folder_path):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
            if total > MAX_UPLOAD_BYTES:
                raise ValueError(
                    f"Upload too large: exceeds {MAX_UPLOAD_BYTES // (1024**2)} MB limit."
                )
    # NOTE: This function raises ValueError if unsafe characters are found in the path.
    return upload_folder_to_hf(
        api=api,
        folder_path=folder_path,
        repo_id=repo_id,
        config_name=config_name,
        split=split,
        submission_name=submission_name,
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
    safe_username = sanitize_path_component(username)
    safe_agent_name = sanitize_path_component(agent_name)

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
            os.makedirs(extracted_dir, exist_ok=True)

            # 2) securely extract only regular files, flatten structure
            # Flatten structure to aid finding the manifest agenteval.json file
            # and because hierarchical structure is not needed
            with tarfile.open(path_to_file.name, "r:gz") as tar:
                for member in tar.getmembers():
                    if not member.isreg():
                        continue
                    fname = os.path.basename(member.name)
                    # skip empty or hidden
                    if not fname or fname.startswith("."):
                        continue
                    fobj = tar.extractfile(member)
                    if not fobj:
                        continue
                    target = os.path.join(extracted_dir, fname)
                    with open(target, "wb") as out:
                        out.write(fobj.read())

            # 3) ensure something was extracted
            if not os.listdir(extracted_dir):
                return format_error("Submission tarball is empty or invalid.")

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
        try:
            checked_upload_folder(
                api=api,
                folder_path=extracted_dir,
                repo_id=SUBMISSION_DATASET,
                config_name=CONFIG_NAME,
                split=val_or_test,
                submission_name=submission_name,
            )
        except ValueError as e:
            return format_error(str(e))

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
        json_path = Path(extracted_dir) / AGENTEVAL_MANIFEST_NAME
        if not json_path.exists():
            return format_error(f"Missing manifest {AGENTEVAL_MANIFEST_NAME}")
        raw = json_path.read_text(encoding="utf-8")
        eval_result = EvalResult.model_validate_json(raw)
        if eval_result.suite_config.version != CONFIG_NAME:
            return format_error(
                f"Error: submitted suite version {eval_result.suite_config.version} "
                f"does not match currently accepted version {CONFIG_NAME}"
            )
        if eval_result.split != val_or_test:
            return format_error(
                f"Error: uploaded split {eval_result.split} does not match selected split {val_or_test}"
            )

        # NOTE: Trusting user-computed scores, but re-computing the derived results based on the log files
        eval_result.results = process_eval_logs(extracted_dir)[0]
        eval_result.save_json(str(json_path))

    except Exception as e:
        return format_error(
            f"Error while scoring the submission: {e}. Be sure to upload a valid submission."
        )

    # # SAVE SCORED SUBMISSION
    if LOCAL_DEBUG:
        print("mock uploaded scored submission")
    else:
        try:
            logs_url_private = checked_upload_folder(
                api=api,
                folder_path=extracted_dir,
                repo_id=SUBMISSION_DATASET,
                config_name=CONFIG_NAME,
                split=val_or_test,
                submission_name=f"{submission_name}_scored",
            )
        except ValueError as e:
            return format_error(str(e))

        # Validation submissions are public for public leaderboard
        if is_validation and not IS_INTERNAL:
            try:
                logs_url_public = checked_upload_folder(
                    api=api,
                    folder_path=extracted_dir,
                    repo_id=SUBMISSION_DATASET_PUBLIC,
                    config_name=CONFIG_NAME,
                    split=val_or_test,
                    submission_name=f"{submission_name}_scored",
                )
            except ValueError as e:
                return format_error(str(e))
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

def refresh():
    # Mock data for validation and test dataframes
    mock_data_val = {
        "Agent": ["Agent A", "Agent B"],
        "Agent description": ["Description A", "Description B"],
        "User/organization": ["User A", "User B"],
        "Submission date": ["2025-01-01", "2025-01-02"],
        "Logs": ["Log A", "Log B"],
        "Score": [85, 90],
        "Cost": [100, 120],
        "tag/code/score": [0.85, 0.90],  # Add mock tag data
        "tag/code/cost": [10, 15],
        "tag/lit/score": [0.75, 0.80],
        "tag/lit/cost": [12, 18],
    }

    mock_data_test = {
        "Agent": ["Agent X", "Agent Y"],
        "Agent description": ["Description X", "Description Y"],
        "User/organization": ["User X", "User Y"],
        "Submission date": ["2025-01-03", "2025-01-04"],
        "Logs": ["Log X", "Log Y"],
        "Score": [75, 80],
        "Cost": [110, 130],
        "tag/code/score": [0.65, 0.70],  # Add mock tag data
        "tag/code/cost": [20, 25],
        "tag/lit/score": [0.60, 0.65],
        "tag/lit/cost": [22, 28],
    }

    # Convert to pandas DataFrames
    eval_dataframe_val = pd.DataFrame(mock_data_val)
    eval_dataframe_test = pd.DataFrame(mock_data_test)

    # Update column types dynamically
    val_col_types = compute_column_types(eval_dataframe_val)
    test_col_types = compute_column_types(eval_dataframe_test)

    return eval_dataframe_val, eval_dataframe_test

# Call refresh() to get mock data
eval_dataframe_val, eval_dataframe_test = refresh()

# Update the Dataframe components with the mock data
leaderboard_table_test = gr.Dataframe(
    value=eval_dataframe_test,
    headers=list(eval_dataframe_test.columns),
    datatype=compute_column_types(eval_dataframe_test),
    interactive=False,
    column_widths=["20%"],
    render=True,
    visible=True,
    elem_classes="table-component",
)

leaderboard_table_val = gr.Dataframe(
    value=eval_dataframe_val,
    headers=list(eval_dataframe_val.columns),
    datatype=compute_column_types(eval_dataframe_val),
    interactive=True,
    column_widths=["20%"],
    render=True,
    visible=False,
elem_classes="table-component",
)


test_col_types = compute_column_types(eval_dataframe_test)
val_col_types = compute_column_types(eval_dataframe_val)

def calculate_frontier(df, cost_col, score_col):
    """Calculate the frontier points based on lowest cost and highest score."""
    # Sort by cost (ascending), then by score (descending) to find the best points
    sorted_df = df.sort_values(by=[cost_col, score_col], ascending=[True, False])

    # Initialize the frontier
    frontier_points = []
    max_score = -float('inf')

    # Iterate through sorted points to find the frontier
    for _, row in sorted_df.iterrows():
        cost, score = row[cost_col], row[score_col]
        # If the score is higher than the previous max, it's a frontier point
        if score > max_score:
            frontier_points.append((cost, score))
            max_score = score

    return pd.DataFrame(frontier_points, columns=[cost_col, score_col])


def create_scatter_plot(df, tag_type):
    """Generate a scatter plot for cost vs. score for the selected tag type."""
    score_col = f"tag/{tag_type}/score"
    cost_col = f"tag/{tag_type}/cost"
    if score_col in df.columns and cost_col in df.columns:
        # Calculate the frontier
        frontier_df = calculate_frontier(df, cost_col, score_col)
        fig = px.scatter(
            df,
            x=cost_col,  # Cost column
            y=score_col,  # Score column
            title=f"Cost vs. Score ({tag_type.capitalize()})",
            labels={cost_col: "Cost (USD)", score_col: "Score"},
        )

        fig.add_trace(
            go.Scatter(
                x=frontier_df[cost_col],
                y=frontier_df[score_col],
                mode="lines+markers",
                line=dict(color="black", dash="dash"),
                marker=dict(color="black", size=10),
                name="Frontier",
            )
        )
        return fig
    return None


theme = gr.themes.Soft(
    primary_hue="pink",
    secondary_hue="teal",
    spacing_size="lg",
).set(
    # background_fill_primary='*secondary_800',
    background_fill_primary_dark='*neutral_100'
)

icon_path = "Ai2_icon_pink_RGB.png"

# Define scatter plot components globally
scatter_plot_test_component = gr.Plot(
    value=None,  # Placeholder value
    label="Cost vs. Score (Test)"
)

scatter_plot_val_component = gr.Plot(
    value=None,  # Placeholder value
    label="Cost vs. Score (Validation)"
)

with gr.Blocks(theme=theme, css=content.css) as demo:
    gr.HTML(TITLE)
    gr.Image(icon_path, elem_id="app-icon", show_label=False, interactive=False, width=100, show_download_button=False, show_fullscreen_button=False)
    gr.Markdown(INTRODUCTION_TEXT, elem_id="markdown-text")
    gr.HTML(INTRO_PARAGRAPH, elem_id="intro-paragraph")

    with gr.Group(elem_classes="unified-container"):
        with gr.Tabs(elem_classes="tab-buttons"):
            # Leaderboard Tab
            with gr.TabItem("🏅 Leaderboard", elem_classes="tab-text"):
                gr.Markdown("### Leaderboard Results")


                # Radio Selector for Results: Test and Results: Validation
                table_selector = gr.Radio(
                    choices=["Results: Test", "Results: Validation"],
                    value="Results: Test",
                    label="Select Table to View",
                    elem_classes="nested-radio-buttons"
                )
                # Define Dataframe components
                leaderboard_table_test = gr.Dataframe(
                    value=eval_dataframe_test,
                    headers=list(eval_dataframe_test.columns),
                    datatype=test_col_types,
                    interactive=False,
                    column_widths=["20%"],
                    render=True,
                    visible=True,
                    elem_classes="table-component"
                )

                leaderboard_table_val = gr.Dataframe(
                    value=eval_dataframe_val,
                    headers=list(eval_dataframe_val.columns),
                    datatype=val_col_types,
                    interactive=False,
                    column_widths=["20%"],
                    render=True,
                    visible=False,
                    elem_classes="table-component"
                )

                # Function to update table visibility
                def update_table(selected_table):
                    if selected_table == "Results: Test":
                        return {
                            leaderboard_table_test: gr.update(visible=True),
                            leaderboard_table_val: gr.update(visible=False),
                        }
                    else:
                        return {
                            leaderboard_table_test: gr.update(visible=False),
                            leaderboard_table_val: gr.update(visible=True),
                        }

                # Update visibility based on selection
                table_selector.change(
                    fn=update_table,
                    inputs=[table_selector],
                    outputs=[leaderboard_table_test, leaderboard_table_val],
                )

            # Submit Your Results Tab
            with gr.TabItem("🚀 Submit Your Results", elem_classes="tab-text"):
                gr.Markdown("### Submit a New Agent for Evaluation")

                with gr.Row():
                    with gr.Column():
                        level_of_test = gr.Radio(
                            ["validation", "test"], value="validation", label="Split"
                        )
                        agent_name_textbox = gr.Textbox(label="Agent name")
                        agent_description_textbox = gr.Textbox(label="Agent description")
                        agent_url_textbox = gr.Textbox(label="URL to agent information")

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


    with gr.Group(elem_classes="unified-container"):
        gr.Markdown("## 📈 Quality vs Cost Analysis", elem_classes="markdown-text")

        # Tag Type Selector
        tag_type_selector = gr.Radio(
            choices=["code", "lit"],
            value="code",
            label="Select Tag Type",
        )

        # Initialize scatter plots with initial values
        with gr.Row():
            scatter_plot_val_component = gr.Plot(value=create_scatter_plot(eval_dataframe_val, "code"), visible=False)
            scatter_plot_test_component = gr.Plot(value=create_scatter_plot(eval_dataframe_test, "code"), visible=True)

        # Update scatter plots dynamically based on tag type
        def update_scatter_plot(tag_type, selected_table):
            is_test_selected = selected_table == "Results: Test"

            # Select appropriate data
            plot_data = eval_dataframe_test if is_test_selected else eval_dataframe_val

            # Update visibility and plot data
            return {
                scatter_plot_test_component: gr.update(visible=is_test_selected, value=create_scatter_plot(plot_data, tag_type)),
                scatter_plot_val_component: gr.update(visible=not is_test_selected, value=create_scatter_plot(plot_data, tag_type)),
            }

        # Trigger scatter plot update when either tag type or table selection changes
        tag_type_selector.change(
            fn=update_scatter_plot,
            inputs=[tag_type_selector, table_selector],
            outputs=[scatter_plot_test_component, scatter_plot_val_component],
        )

        table_selector.change(
            fn=update_scatter_plot,
            inputs=[tag_type_selector, table_selector],
            outputs=[scatter_plot_test_component, scatter_plot_val_component],
        )

scheduler = BackgroundScheduler()
scheduler.add_job(restart_space, "interval", seconds=3600)
scheduler.start()
if LOCAL_DEBUG:
    demo.launch(debug=True)
else:
    demo.launch(server_name="0.0.0.0", server_port=7860, debug=True)
