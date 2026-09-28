"""PostToolUse hook: lint Python files changed by a Codex apply_patch call."""

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PATCH_PATH = re.compile(r"^\*\*\* (?:Add File|Update File|Move to): (.+)$", re.MULTILINE)


def executable(name: str) -> str | None:
    local = ROOT / ".venv" / "bin" / name
    return str(local) if local.is_file() else shutil.which(name)


def main() -> int:
    try:
        event = json.load(sys.stdin)
        patch = event.get("tool_input", {}).get("command", "")
    except (ValueError, AttributeError, TypeError):
        return 0
    if not isinstance(patch, str):
        return 0

    ruff = executable("ruff")
    mypy = executable("mypy")
    failures: list[str] = []
    for raw_path in sorted(set(PATCH_PATH.findall(patch))):
        path = (ROOT / raw_path).resolve()
        try:
            relative = path.relative_to(ROOT)
        except ValueError:
            continue
        if not path.is_file() or path.suffix != ".py":
            continue
        if relative.parts[0] not in {"rag", "tests", "scripts"}:
            continue

        checks = [(ruff, ["check", "--force-exclude", str(relative)])]
        if relative.parts[0] == "rag":
            checks.append((mypy, ["--ignore-missing-imports", str(relative)]))
        for program, args in checks:
            if program is None:
                continue
            result = subprocess.run(
                [program, *args], cwd=ROOT, capture_output=True, text=True, check=False
            )
            if result.returncode:
                failures.append(
                    f"{Path(program).name} failed for {relative}:\n"
                    f"{result.stdout}{result.stderr}"
                )

    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
