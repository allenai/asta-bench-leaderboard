---
title: AstaBench Leaderboard
emoji: 🥇
colorFrom: green
colorTo: indigo
sdk: docker
app_file: app.py
pinned: true
license: apache-2.0
hf_oauth: true
app_port: 7860
failure_strategy: none
tags:
  - leaderboard
---

## Development
The leaderboard is built using the [HuggingFace Datasets](https://huggingface.co/docs/datasets/index) library, which provides a convenient way to manage and query datasets.
It's currently pointed at the [AstaBench Leaderboard](https://huggingface.co/datasets/allenai/asta-bench-internal-results/) dataset, which is a public dataset hosted on HuggingFace.

To run the leaderboard locally first make sure to set this env variable:
```bash
export IS_INTERNAL=true
```
You can then start it up with the following command:
```bash
python app.py
```
This will start a local server that you can access in your web browser at `http://localhost:7860`.

## Hugging Face Integration
The repo backs two Hugging Face leaderboard spaces:
- https://huggingface.co/spaces/allenai/asta-bench-internal-leaderboard
- https://huggingface.co/spaces/allenai/asta-bench-leaderboard

Please follow the steps below to push changes to the leaderboards on Hugging Face.

Before pushing, make sure to merge your changes to the `main` branch of this repository. (following the standard GitHub workflow of creating a branch, making changes, and then merging it back to `main`).

Before pushing for the first time, you'll need to add the Hugging Face remote repositories if you haven't done so already. You can do this by running the following command:

```bash
git remote add huggingface https://huggingface.co/spaces/allenai/asta-bench-internal-leaderboard
git remote add huggingface-public https://huggingface.co/spaces/allenai/asta-bench-leaderboard
```
You can verify that the remotes have been added by running:

```bash
git remote -v
```
Then, to push the changes to the Hugging Face leaderboards, you can use the following commands:

```bash
git push huggingface main:main   
git push huggingface-public main:main
```
