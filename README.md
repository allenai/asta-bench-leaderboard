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

To run the leaderboard locally, you can use the following command:

```bash
python app.py
```
This will start a local server that you can access in your web browser at `http://localhost:7860`.

## Hugging Face Integration
This repo is already integrated with Hugging Face, please follow the steps below to push changes to the leaderboard on Hugging Face.
First make sure to merge your changes to the `main` branch of this repository. (following the standard GitHub workflow of creating a branch, making changes, and then merging it back to `main`).
Then, to push the changes to the Hugging Face leaderboard, you can use the following command:

```bash
git push huggingface main:main   
```
