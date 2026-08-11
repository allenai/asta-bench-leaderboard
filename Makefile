.PHONY: help version set-version push-version-tag deploy-internal check-version

PY := python

help:
	@echo "AstaBench leaderboard release targets:"
	@echo "  version           Show the current version (from VERSION)"
	@echo "  set-version       Set the version (requires VERSION=x.y.z)"
	@echo "  check-version     Validate the VERSION file is well-formed semver"
	@echo "  push-version-tag  Tag the current commit vX.Y.Z and push it (fires the public release workflow)"
	@echo "  deploy-internal   Manual fallback: mirror the current HEAD to the internal Space (needs HF_TOKEN)"

version:
	@$(PY) scripts/manage-version.py show

check-version:
	@$(PY) scripts/manage-version.py check

# Bump the version. Usage: make set-version VERSION=1.2.3
set-version:
	@$(PY) scripts/manage-version.py set $(VERSION)

# Tag the current (merged) commit with vX.Y.Z from the VERSION file and push the
# tag. Pushing a v* tag triggers .github/workflows/release-public.yml, which
# mirrors the tagged commit to the public Space. Run this from an up-to-date main.
push-version-tag:
	@$(PY) scripts/manage-version.py check
	@V=$$($(PY) scripts/manage-version.py show); \
	git tag v$$V && git push origin v$$V && echo "Pushed tag v$$V"

# Manual fallback for the internal Space (CI normally does this on push to main).
# Requires HF_TOKEN with write access to the internal Space.
deploy-internal:
	@test -n "$$HF_TOKEN" || { echo "HF_TOKEN is required"; exit 1; }
	git push --force "https://hf:$$HF_TOKEN@huggingface.co/spaces/allenai/asta-bench-internal-leaderboard" HEAD:main
