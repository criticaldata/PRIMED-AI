#!/usr/bin/env python3
"""Create GitHub issues for PRIMED-AI tasks from TASK.md and update issue links."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASK_MD = ROOT / "TASK.md"
MAPPING_PATH = ROOT / "scripts" / "issue_mapping.json"
REPO = "criticaldata/PRIMED-AI"

TASK_HEADER = re.compile(r"^#### (?P<id>[A-Z]\d{2}) — (?P<title>.+)$")
INDEX_ROW = re.compile(
    r"^\| \*\*(?P<id>[A-Z]\d{2})\*\* \| (?:TBD|\[#\d+\]\([^)]+\)) \| (?P<category>[^|]+) \| (?P<title>[^|]+) \| (?P<priority>[^|]+) \| (?P<complexity>[^|]+) \| (?P<est_days>\d+) \| — \| (?P<refs>[^|]+) \|$"
)
FIELD_ROW = re.compile(r"^\| \*\*(?P<key>[^*]+)\*\* \| (?P<value>.+) \|$")


def gh_bin() -> str:
    candidates = [
        ROOT / ".tools" / "gh_2.94.0_macOS_arm64" / "bin" / "gh",
        Path("/Users/buno/Documents/coding/criticaldata/RecursiveJEPA/.tools/gh_2.93.0_macOS_arm64/bin/gh"),
        Path("gh"),
    ]
    for candidate in candidates:
        if candidate == Path("gh") or candidate.is_file():
            return str(candidate)
    return "gh"


def get_token() -> str:
    if os.environ.get("GH_TOKEN"):
        return os.environ["GH_TOKEN"]
    if os.environ.get("GITHUB_TOKEN"):
        return os.environ["GITHUB_TOKEN"]
    proc = subprocess.run(
        ["git", "credential", "fill"],
        input="protocol=https\nhost=github.com\n",
        capture_output=True,
        text=True,
        check=True,
    )
    for line in proc.stdout.splitlines():
        if line.startswith("password="):
            return line.split("=", 1)[1]
    raise RuntimeError("Could not obtain GitHub token from git credential helper")


def parse_refs(raw: str) -> list[str]:
    raw = raw.strip()
    if raw in {"—", "-", ""}:
        return []
    return [part.strip() for part in raw.split(",")]


def parse_tasks(text: str) -> list[dict]:
    index: dict[str, dict] = {}
    for line in text.splitlines():
        match = INDEX_ROW.match(line)
        if match:
            task_id = match.group("id")
            index[task_id] = {
                "id": task_id,
                "category": match.group("category").strip(),
                "title": match.group("title").strip(),
                "priority": match.group("priority").strip().strip("*"),
                "complexity": match.group("complexity").strip(),
                "est_days": int(match.group("est_days")),
                "refs": parse_refs(match.group("refs")),
            }

    sections = re.split(r"\n(?=#### [A-Z]\d{2} — )", text)
    details: dict[str, dict] = {}
    for section in sections:
        lines = section.splitlines()
        if not lines:
            continue
        header = TASK_HEADER.match(lines[0])
        if not header:
            continue
        task_id = header.group("id")
        fields: dict[str, str] = {}
        for line in section.splitlines():
            field_match = FIELD_ROW.match(line)
            if field_match:
                fields[field_match.group("key")] = field_match.group("value").strip()

        body_start = section.find("**Description**")
        description = section[body_start:].split("\n---", 1)[0].strip() if body_start >= 0 else ""
        if description.startswith("**Description**"):
            description = description[len("**Description**") :].strip()

        details[task_id] = {
            "features": fields.get("Features / method", ""),
            "description": description,
        }

    tasks: list[dict] = []
    for task_id in sorted(index, key=lambda x: (x[0], int(x[1:]))):
        meta = index[task_id]
        detail = details.get(task_id, {"features": "", "description": ""})
        tasks.append({**meta, **detail})
    return tasks


def existing_issue_map(token: str) -> dict[str, dict]:
    env = {**os.environ, "GH_TOKEN": token}
    proc = subprocess.run(
        [
            gh_bin(),
            "issue",
            "list",
            "--repo",
            REPO,
            "--state",
            "all",
            "--limit",
            "200",
            "--json",
            "number,title,url",
        ],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )
    found: dict[str, dict] = {}
    for issue in json.loads(proc.stdout):
        title = issue["title"]
        if title.startswith("[") and "]" in title:
            task_id = title[1 : title.index("]")]
            found[task_id] = {"number": issue["number"], "url": issue["url"]}
    return found


def update_issue(token: str, number: int, body: str) -> None:
    env = {**os.environ, "GH_TOKEN": token}
    proc = subprocess.run(
        [
            gh_bin(),
            "api",
            f"repos/{REPO}/issues/{number}",
            "--method",
            "PATCH",
            "--input",
            "-",
        ],
        input=json.dumps({"body": body}),
        capture_output=True,
        text=True,
        env=env,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr or proc.stdout)


def create_issue(token: str, title: str, body: str) -> dict:
    env = {**os.environ, "GH_TOKEN": token}
    for attempt in range(6):
        proc = subprocess.run(
            [
                gh_bin(),
                "api",
                f"repos/{REPO}/issues",
                "--method",
                "POST",
                "--input",
                "-",
            ],
            input=json.dumps({"title": title, "body": body}),
            capture_output=True,
            text=True,
            env=env,
        )
        if proc.returncode == 0:
            created = json.loads(proc.stdout)
            return {"number": created["number"], "url": created["html_url"]}
        if attempt < 5:
            time.sleep(2**attempt)
            continue
        raise RuntimeError(proc.stderr or proc.stdout)
    raise RuntimeError("Failed to create issue")


def issue_body(task: dict) -> str:
    refs = ", ".join(task["refs"]) if task["refs"] else "—"
    return f"""## Task metadata

| Field | Value |
|-------|-------|
| **Task ID** | `{task['id']}` |
| **Category** | {task['category']} |
| **Priority** | {task['priority']} |
| **Complexity** | {task['complexity']} |
| **Est. days** | {task['est_days']} *(~1–2 h/day contributor pace)* |
| **Key references** | {refs} |

## Features / method

{task['features']}

## Description

{task['description']}

---
*Tracked in [TASK.md](https://github.com/{REPO}/blob/main/TASK.md).*
"""


def update_task_md(mapping: list[dict]) -> None:
    text = TASK_MD.read_text()
    by_id = {task["id"]: task for task in mapping}

    for task_id, task in by_id.items():
        issue_num = task["github_issue_id"]
        issue_url = task["github_issue_url"]
        issue_link = f"[#{issue_num}]({issue_url})"

        text = re.sub(
            rf"(\| \*\*{task_id}\*\* \|) TBD (\|)",
            rf"\1 {issue_link} \2",
            text,
            count=1,
        )

        section_pattern = (
            rf"(#### {task_id} —[\s\S]*?\| \*\*GitHub Issue\*\* \|)"
            r" (?:TBD|\[#\d+\]\([^)]+\)) (\|)"
        )
        text = re.sub(section_pattern, rf"\1 {issue_link} \2", text, count=1)

    category_prefix = {
        "Infrastructure": "I",
        "Data": "D",
        "Model": "M",
        "Evaluation": "E",
        "Writing": "W",
    }
    for category in category_prefix:
        cat_tasks = sorted(
            [t for t in mapping if t["category"] == category], key=lambda t: t["id"]
        )
        if not cat_tasks:
            continue
        link_line = " · ".join(
            f"[{t['id']}]({t['github_issue_url']})" for t in cat_tasks
        )
        old_line = " · ".join(t["id"] for t in cat_tasks)
        text = text.replace(
            f"**{len(cat_tasks)} tasks** · {old_line}",
            f"**{len(cat_tasks)} tasks** · {link_line}",
            1,
        )

    text = text.replace(
        "> **GitHub issues:** Create one issue per task ID when onboarding. Replace `TBD` in the task index with issue links.",
        f"> **GitHub issues:** {len(mapping)} issues created — see task index and category sections below.",
    )

    TASK_MD.write_text(text)


def main() -> None:
    tasks = parse_tasks(TASK_MD.read_text())
    if not tasks:
        raise SystemExit("No tasks parsed from TASK.md")

    token = get_token()
    existing = existing_issue_map(token)
    mapping: list[dict] = []

    for task in tasks:
        title = f"[{task['id']}] {task['title']}"
        body = issue_body(task)
        if task["id"] in existing:
            created = existing[task["id"]]
            update_issue(token, created["number"], body)
            print(f"Updated #{created['number']}: {task['id']}", file=sys.stderr)
        else:
            created = create_issue(token, title, body)
            print(f"Created #{created['number']}: {task['id']}", file=sys.stderr)
            time.sleep(1)
        mapping.append(
            {
                **task,
                "github_issue_id": created["number"],
                "github_issue_url": created["url"],
            }
        )

    MAPPING_PATH.write_text(json.dumps(mapping, indent=2) + "\n")
    update_task_md(mapping)
    print(f"Wrote {MAPPING_PATH}", file=sys.stderr)
    print(f"Updated {TASK_MD}", file=sys.stderr)


if __name__ == "__main__":
    main()
