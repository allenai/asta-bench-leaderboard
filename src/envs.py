import os

from huggingface_hub import HfApi

# Info to change for your repository
# ----------------------------------
TOKEN = os.environ.get("HF_TOKEN") # A read/write token for your org

OWNER = "allenai"
PROJECT_NAME = "nora-bench"
# ----------------------------------

REPO_ID = os.environ.get("REPO_ID", f"{OWNER}/{PROJECT_NAME}-leaderboard")
QUEUE_REPO = f"{OWNER}/{PROJECT_NAME}-requests"
RESULTS_REPO = os.environ.get("RESULTS_REPO", f"{OWNER}/{PROJECT_NAME}-results")

# If you setup a cache later, just change HF_HOME
CACHE_PATH=os.getenv("HF_HOME", ".")

# Local caches
EVAL_REQUESTS_PATH = os.path.join(CACHE_PATH, "eval-queue")
EVAL_RESULTS_PATH = os.path.join(CACHE_PATH, "eval-results")
EVAL_REQUESTS_PATH_BACKEND = os.path.join(CACHE_PATH, "eval-queue-bk")
EVAL_RESULTS_PATH_BACKEND = os.path.join(CACHE_PATH, "eval-results-bk")

API = HfApi(token=TOKEN)
