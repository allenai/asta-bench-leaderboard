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

Deployment is automated via GitHub Actions and follows a two-track model,
inspired by [`allenai/asta-plugins`](https://github.com/allenai/asta-plugins):
the **internal** Space tracks `main` continuously, and the **public** Space is
promoted deliberately on a version tag.

### Internal Space — tracks `main` (automatic)

Every push to `main` triggers [`.github/workflows/deploy-internal.yml`](.github/workflows/deploy-internal.yml),
which mirrors the commit to the internal Space (`allenai/asta-bench-internal-leaderboard`),
which then auto-rebuilds. So merging a PR to `main` is all it takes to update
internal — it's the always-current staging environment. No manual push needed.

### Public Space — promoted on a version tag (deliberate)

The public Space (`allenai/asta-bench-leaderboard`) only updates when you cut a
release, so it never receives anything that hasn't already soaked on internal:

```bash
# 1. Bump the version on a branch, open a PR, merge it to main.
make set-version VERSION=0.2.0
#    ... commit, PR, review, merge ...

# 2. From an up-to-date main, tag the merged commit and push the tag.
git checkout main && git pull
make push-version-tag           # tags v0.2.0 and pushes it
```

Pushing the `v0.2.0` tag triggers [`.github/workflows/release-public.yml`](.github/workflows/release-public.yml),
which verifies the tag matches the `VERSION` file and mirrors *that exact tagged
commit* to the public Space. The tag is the source of truth for what's public.

`VERSION` is the leaderboard **app** version (semver `x.y.z`); it is distinct
from `HF_CONFIG` (e.g. `1.0.0`), which versions the results dataset. The running
app prints its version at launch (visible in the Space logs) so you can confirm
which release is live.

### Credentials and gating

The two deploy jobs run under the `internal` and `public`
[GitHub Environments](https://docs.github.com/en/actions/deployment/targeting-different-environments)
and use separate secrets — `HF_TOKEN` (write to the internal Space) and
`HF_PUBLIC_TOKEN` (write to the public Space) — so public-write scope is granted
only to the tag-gated release job. Add required reviewers to the `public`
environment to make promotion a human-approved step.

### Manual fallback

If you ever need to push by hand (CI outage, first-time setup), add the remotes:

```bash
git remote add huggingface https://huggingface.co/spaces/allenai/asta-bench-internal-leaderboard
git remote add huggingface-public https://huggingface.co/spaces/allenai/asta-bench-leaderboard
git push huggingface main:main          # internal
git push huggingface-public main:main   # public
```
