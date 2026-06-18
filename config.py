import os

LOCAL_DEBUG = not (os.environ.get("system") == "spaces")
CONFIG_NAME = os.getenv("HF_CONFIG", "1.0.0-dev1") # This corresponds to 'config' in LeaderboardViewer
IS_INTERNAL = os.environ.get("IS_INTERNAL", "false").lower() == "true"

# Put all contact info in the same dataset so there's only one place to search. This is private to Ai2.
CONTACT_DATASET = os.getenv("CONTACT_DATASET", "allenai/asta-bench-internal-contact-info")

# Allow env var overrides for dev/testing
SUBMISSION_DATASET = os.getenv("SUBMISSION_DATASET")
RESULTS_DATASET = os.getenv("RESULTS_DATASET")
LEADERBOARD_PATH = os.getenv("LEADERBOARD_PATH")

if not SUBMISSION_DATASET or not RESULTS_DATASET or not LEADERBOARD_PATH:
    if IS_INTERNAL:
        # datasets backing the internal leaderboard
        SUBMISSION_DATASET = SUBMISSION_DATASET or "allenai/asta-bench-internal-submissions"
        RESULTS_DATASET = RESULTS_DATASET or "allenai/asta-bench-internal-results"
        LEADERBOARD_PATH = LEADERBOARD_PATH or "allenai/asta-bench-internal-leaderboard"
    else:
        # datasets backing the public leaderboard
        SUBMISSION_DATASET = SUBMISSION_DATASET or "allenai/asta-bench-submissions"
        RESULTS_DATASET = RESULTS_DATASET or "allenai/asta-bench-results"
        LEADERBOARD_PATH = LEADERBOARD_PATH or "allenai/asta-bench-leaderboard"

DATA_DIR = "/tmp/abl/data/" + CONFIG_NAME
EXTRACTED_DATA_DIR = os.path.join(DATA_DIR, "extracted")
