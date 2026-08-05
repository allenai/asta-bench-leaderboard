# Presence of a conftest.py at the repo root makes pytest treat this directory
# as the rootdir and prepend it to sys.path (default "prepend" import mode).
# The tests under tests/ import the app's top-level modules (aliases, config,
# submission, ...) which live here at the repo root, so they must be importable
# when CI runs `pytest tests/integration/` as the console script (which, unlike
# `python -m pytest`, does not add the current directory to sys.path).
