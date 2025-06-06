# app.py
import gradio as gr
import os

from apscheduler.schedulers.background import BackgroundScheduler
from huggingface_hub import HfApi
import literature_understanding, main_page, c_and_e, data_analysis, e2e

from content import TITLE

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
with demo.route("Literature Understanding"):
    literature_understanding.demo.render()
with demo.route("Code & Execution"):
    c_and_e.demo.render()
with demo.route("Data Analysis"):
    data_analysis.demo.render()
with demo.route("End to End"):
    e2e.demo.render()

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

