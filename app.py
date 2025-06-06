# app.py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np # For mock data generation

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
# import numpy as np # May not be needed directly here if LeaderboardViewer handles it
import pandas as pd
import requests
from agenteval import (
    # compute_summary_statistics, # This will now be used by LeaderboardViewer
    process_eval_logs,
    upload_folder_to_hf,
    upload_summary_to_hf,
)
from agenteval.models import EvalResult # Used by submission and LeaderboardViewer (implicitly)
from agenteval.upload import sanitize_path_component
from apscheduler.schedulers.background import BackgroundScheduler
from datasets import Dataset, DatasetDict, VerificationMode, load_dataset # load_dataset used by LV
from datasets.data_files import EmptyDatasetError
from huggingface_hub import HfApi
import literature_understanding, main_page

from leaderboard_viewer import LeaderboardViewer

from content import (
    CITATION_BUTTON_LABEL,
    CITATION_BUTTON_TEXT,
    INTRODUCTION_TEXT,
    SUBMISSION_TEXT,
    TITLE,
    INTRO_PARAGRAPH,
    format_error,
    format_log,
    format_warning,
    hf_uri_to_web_url, # LeaderboardViewer might need this or similar for log links
    hyperlink, # LeaderboardViewer might need this or similar for log links
)

# --- Constants and Configuration  ---
USE_MOCK_DATA = True
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



# --- Theme Definition ---
theme = gr.themes.Soft(
    primary_hue="pink",
    secondary_hue="teal",
    spacing_size="lg",
).set(
    background_fill_primary_dark='*neutral_100'
)
# --- Gradio App Definition ---
demo = gr.Blocks(theme=theme)
with demo:
    gr.HTML(TITLE)
    main_page.demo.render()
with demo.route("Literature_Understanding"):
    literature_understanding.demo.render()

# --- Scheduler and Launch
def restart_space_job():
    print("Scheduler: Attempting to restart space.")
    try:
        api.restart_space(repo_id=LEADERBOARD_PATH)
        print("Scheduler: Space restart request sent.")
    except Exception as e:
        print(f"Scheduler: Error restarting space: {e}")
    scheduler = BackgroundScheduler(timezone="UTC")
    scheduler.add_job(restart_space_job, "interval", hours=1)
    scheduler.start()


# Launch the Gradio app
if __name__ == "__main__":
    if LOCAL_DEBUG:
        print("Launching in LOCAL_DEBUG mode.")
        def get_initial_global_tag_choices(): return ["Overall", "TagA"]
        demo.launch(debug=True)
    else:
        print("Launching in Space mode.")
        # For Spaces, share=False is typical unless specific tunneling is needed.
        # debug=True can be set to False for a "production" Space.
        demo.launch(server_name="0.0.0.0", server_port=7860, debug=True, share=False)

