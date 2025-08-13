import os

LOCAL_DEBUG = not (os.environ.get("system") == "spaces")
CONFIG_NAME = os.getenv("HF_CONFIG", "1.0.0-dev1") # This corresponds to 'config' in LeaderboardViewer
IS_INTERNAL = os.environ.get("IS_INTERNAL", "false").lower() == "true"

DATASET_PREFIX = os.getenv("ABL_DATASET_PREFIX", "allenai")
PROJECT_NAME = "asta-bench" + ("-internal" if IS_INTERNAL else "")
SUBMISSION_DATASET = f"{DATASET_PREFIX}/{PROJECT_NAME}-submissions"
SUBMISSION_DATASET_PUBLIC = f"{DATASET_PREFIX}/{PROJECT_NAME}-submissions-public"
CONTACT_DATASET = f"{DATASET_PREFIX}/{PROJECT_NAME}-contact-info"
RESULTS_DATASET = f"{DATASET_PREFIX}/{PROJECT_NAME}-results" # This is the repo_id for LeaderboardViewer
LEADERBOARD_PATH = f"{DATASET_PREFIX}/{PROJECT_NAME}-leaderboard"

if LOCAL_DEBUG:
    DATA_DIR = os.path.join(os.path.dirname(__file__), "data", CONFIG_NAME)
else:
    DATA_DIR = "/home/user/data/" + CONFIG_NAME
EXTRACTED_DATA_DIR = os.path.join(DATA_DIR, "extracted")
