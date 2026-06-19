import json
import subprocess
from datetime import datetime
from pathlib import Path


def get_git_sha() -> str:
    """Return the current git commit SHA, or 'unknown' if unavailable."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def create_run_dir(base_dir: str | Path) -> Path:
    """Create a timestamped run directory."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(base_dir) / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_run_metadata(run_dir: str | Path, metadata: dict) -> None:
    """Save run metadata, including the current git SHA."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        **metadata,
        "git_sha": get_git_sha(),
    }

    with open(run_dir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)