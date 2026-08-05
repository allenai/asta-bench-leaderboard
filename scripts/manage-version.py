#!/usr/bin/env python3
"""Version management for the AstaBench leaderboard.

The leaderboard is a HuggingFace Docker Space with a single source of truth for
its release version: the top-level ``VERSION`` file (semver ``x.y.z``). This is
the *leaderboard app* version and is intentionally distinct from the
results-dataset config version (``HF_CONFIG``, e.g. ``1.0.0``).

Commands:
  show          Print the current version.
  check         Validate that VERSION is well-formed semver (exit non-zero if not).
  check-tag T   Validate that git tag ``T`` (``vX.Y.Z``) matches VERSION.
  set X.Y.Z     Write a new version to the VERSION file.

The release flow is: ``set`` the version on a PR -> merge to main -> tag
``vX.Y.Z`` -> the release workflow mirrors the tagged commit to the public Space.
"""

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VERSION_FILE = PROJECT_ROOT / "VERSION"

SEMVER = re.compile(r"^\d+\.\d+\.\d+$")


def get_version() -> str:
    return VERSION_FILE.read_text().strip()


def set_version(new_version: str) -> int:
    if not SEMVER.match(new_version):
        print(f"Error: version must be semver x.y.z (got {new_version!r})", file=sys.stderr)
        return 1
    VERSION_FILE.write_text(new_version + "\n")
    print(f"VERSION set to {new_version}")
    print("Next: commit, open a PR, merge to main, then `make push-version-tag`.")
    return 0


def check() -> int:
    version = get_version()
    if not SEMVER.match(version):
        print(f"Error: VERSION file is not valid semver: {version!r}", file=sys.stderr)
        return 1
    print(f"VERSION ok: {version}")
    return 0


def check_tag(tag: str) -> int:
    if check() != 0:
        return 1
    expected = f"v{get_version()}"
    if tag != expected:
        print(
            f"Error: tag {tag!r} does not match VERSION file (expected {expected!r}).\n"
            "Bump VERSION with `make set-version VERSION=x.y.z` before tagging.",
            file=sys.stderr,
        )
        return 1
    print(f"Tag {tag} matches VERSION.")
    return 0


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    cmd = sys.argv[1]
    if cmd == "show":
        print(get_version())
        return 0
    if cmd == "check":
        return check()
    if cmd == "check-tag":
        if len(sys.argv) < 3:
            print("Usage: manage-version.py check-tag vX.Y.Z", file=sys.stderr)
            return 1
        return check_tag(sys.argv[2])
    if cmd == "set":
        if len(sys.argv) < 3:
            print("Usage: manage-version.py set X.Y.Z", file=sys.stderr)
            return 1
        return set_version(sys.argv[2])
    print(f"Unknown command: {cmd!r}", file=sys.stderr)
    print(__doc__)
    return 1


if __name__ == "__main__":
    sys.exit(main())
