import os

LOCAL_DEBUG = not (os.environ.get("system") == "spaces")
CONFIG_NAME = os.getenv("HF_CONFIG", "1.0.0-dev1") # This corresponds to 'config' in LeaderboardViewer
IS_INTERNAL = "true"

# Put all contact info in the same dataset so there's only one place to search. This is private to Ai2.
CONTACT_DATASET = f"allenai/asta-bench-internal-contact-info"

if IS_INTERNAL:
    # datasets backing the internal leaderboard
    SUBMISSION_DATASET = f"allenai/asta-bench-internal-submissions"
    RESULTS_DATASET = f"allenai/asta-bench-internal-results"
    LEADERBOARD_PATH = f"allenai/asta-bench-internal-leaderboard"
else:
    # datasets backing the public leaderboard
    SUBMISSION_DATASET = f"allenai/asta-bench-submissions"
    RESULTS_DATASET = f"allenai/asta-bench-results"
    LEADERBOARD_PATH = f"allenai/asta-bench-leaderboard"

DATA_DIR = "/tmp/abl/data/" + CONFIG_NAME
EXTRACTED_DATA_DIR = os.path.join(DATA_DIR, "extracted")
