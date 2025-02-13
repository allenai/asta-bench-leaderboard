---
title: NoraBench Leaderboard
emoji: 🥇
colorFrom: green
colorTo: indigo
sdk: gradio
app_file: app.py
pinned: true
license: apache-2.0
---
# Norabench leaderboard

This is a fork of https://huggingface.co/spaces/demo-leaderboard-backend/leaderboard

## Changes

- Replaced Models with Solvers (model characteristics commented out, to be replaced with solver characteristics).
- Disabled mechanisms to allow user submissions via the website.
- Added new type of results, called `results_cost`, which stores costs. Costs are treated as separate tasks in the application logic (tasks defined in `src/about.py` have a new `is_cost` property).
- Vendored in Norabench utilities for parsing result objects and computing costs.

## Deploying

To deploy changes, push to HuggingFace remote (internal leaderboard at `https://huggingface.co/spaces/allenai/nora-bench-leaderboard-internal`).

# Start the configuration

Most of the variables to change for a default leaderboard are in `src/env.py` (replace the path for your leaderboard) and `src/about.py` (for tasks).

Results files should follow the `ResultSet` type defined in `norabench.suite.score` (vendored here as `src/vendor/norabench/suite/score.py`).

Request files are created automatically by this tool.

If you encounter problem on the space, don't hesitate to restart it to remove the create eval-queue, eval-queue-bk, eval-results and eval-results-bk created folder.
Regardless, the app restarts on interval set in `src/app.py`.

# Code logic for more complex edits

You'll find 
- the main table' columns names and properties in `src/display/utils.py`
- the logic to read all results and request files, then convert them in dataframe lines, in `src/leaderboard/read_evals.py`, and `src/populate.py`. `read_evals.EvalResult.init_from_json_file()` processes a results file and `read_evals.EvalResult.to_dict()` computes overall score and cost (aggregated across tasks).
- (disabled) the logic to allow or filter submissions in `src/submission/submit.py` and `src/submission/check_validity.py`