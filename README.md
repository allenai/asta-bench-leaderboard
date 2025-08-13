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

To run the leaderboard locally, you will first need to clone all the datasets that the leaderboard depends on.

Set up SSH-based access to HuggingFace, exactly as you would for GitHub. https://huggingface.co/settings/keys
Often this means copy the contents of ~/.ssh/id_rsa.pub to your HuggingFace account. Check that HF recognizes
you over SSH by briefly connecting to it. You should see a message like this:

```
$> ssh git@hf.co
PTY allocation request failed on channel 0
Hi dirkraft, welcome to Hugging Face.
Connection to hf.co closed.
```

To avoid polluting the real datasets on HuggingFace, clone all the datasets into a local directory.
You will set this as ABL_DATASET_PREFIX for development. Then you will be able to freely manipulate each
of these locally to work on the leaderboard and avoid generating a lot of junk commits on HuggingFace.

Internal datasets
* `git clone git@hf.co:datasets/allenai/asta-bench-internal-submissions`
* `git clone git@hf.co:datasets/allenai/asta-bench-internal-results`
* `git clone git@hf.co:datasets/allenai/asta-bench-internal-contact-info`

Public datasets
* `git clone git@hf.co:datasets/allenai/asta-bench-submissions`
* `git clone git@hf.co:datasets/allenai/asta-bench-results`
* `git clone git@hf.co:datasets/allenai/asta-bench-contact-info`

```bash
export ABL_DATASET_PREFIX="/path/to/dataset/root"
python app.py
```
This will start a local server that you can access in your web browser at `http://localhost:7860`.

## Hugging Face Integration
This repo is already integrated with Hugging Face, please follow the steps below to push changes to the leaderboard on Hugging Face.
First make sure to merge your changes to the `main` branch of this repository. (following the standard GitHub workflow of creating a branch, making changes, and then merging it back to `main`).
First you need to add the Hugging Face remote repository if you haven't done so already. You can do this by running the following command:

```bash
git remote add huggingface https://huggingface.co/spaces/allenai/asta-bench-internal-leaderboard
```
You can verify that the remote has been added by running:

```bash
git remote -v
```
Then, to push the changes to the Hugging Face leaderboard, you can use the following command:

```bash
git push huggingface main:main   
```
