"""Per-trajectory features: action-type counts, edit churn, patch shape.

Bash command classifier, regex-based. Categories tuned by spot-checking
~30 commands; not validated against hand labels.
"""
from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from traj_metrics import role_of

ACTIONS = ("read", "edit", "scratch_write", "search", "test", "git", "install", "other")

# Verbs we treat as read-only file inspection.
_READ = re.compile(
    r"\b(?:cat|head|tail|less|more|file|wc|nl|hexdump)\b"
    r"|\bsed\s+-[a-zA-Z]*n\b"  # sed -n is print-only
)
_SEARCH = re.compile(r"\b(?:find|grep|rg|ag|ack|locate|fd|tree)\b|\bls\b")
_TEST = re.compile(
    r"\b(?:python\d?|py|pytest|tox|nosetests|unittest|node|npm\s+test|make\s+test)\b"
)
_GIT = re.compile(r"\bgit\s+\w+")
_INSTALL = re.compile(r"\bpip\s+(?:install|uninstall)\b|\bapt(?:-get)?\s+(?:install|update)\b|\bconda\s+install\b")

# Edit detection: in-place sed, heredoc redirect to a path, append/overwrite
# redirect, apply_patch, python-side .write_text/write calls.
_SED_INPLACE = re.compile(r"\bsed\s+-[a-zA-Z]*i")
_HEREDOC = re.compile(r"<<-?\s*['\"]?\w+['\"]?")
_REDIRECT = re.compile(r"(?<![&|>])\s>>?\s*(?!&|/dev/null)")
_APPLY_PATCH = re.compile(r"\bapply_patch\b|\bgit\s+apply\b")
_PY_WRITE = re.compile(r"\.write_text\s*\(|\.write\s*\(|open\([^)]*['\"]w['\"]\)")

# Extract the path being written by simple commands. Catches:
#   sed -i ... /path/to/file
#   cat <<EOF > /path/to/file
#   echo ... > /path/to/file
#   apply_patch /path/to/file
# Misses obscure patterns; that's fine, we treat the path as None then.
_PATH_AFTER_REDIR = re.compile(r">>?\s*([^\s|;<>&]+)")
_PATH_LAST_ARG = re.compile(r"\b([/.][\w./_\-]+\.[A-Za-z0-9]+)(?:\s|$)")


def classify(cmd: str) -> str:
    if not cmd:
        return "other"
    # Order matters: install before test before search, edit checks first
    # because edits often look like reads with a redirect tacked on.
    if _SED_INPLACE.search(cmd) or _HEREDOC.search(cmd) and _REDIRECT.search(cmd):
        return "edit"
    if _APPLY_PATCH.search(cmd) or _PY_WRITE.search(cmd):
        return "edit"
    # `> path` to a repo file is an edit; to /tmp/scratch.txt or out.log it
    # is a scratch write. Earlier the latter was bucketed as edit and
    # n_edit picked up agents using `> patch.txt` as a clipboard, which
    # is what the prompt repeatedly asks them to do.
    if _REDIRECT.search(cmd) and "/dev/null" not in cmd:
        target = _redirect_target(cmd)
        if target and _is_repo_path(target):
            return "edit"
        return "scratch_write"
    if _INSTALL.search(cmd):
        return "install"
    if _GIT.search(cmd):
        return "git"
    if _TEST.search(cmd):
        return "test"
    if _SEARCH.search(cmd):
        return "search"
    if _READ.search(cmd):
        return "read"
    return "other"


def _redirect_target(cmd: str) -> str | None:
    m = _PATH_AFTER_REDIR.search(cmd)
    return _normalise(m.group(1)) if m else None


def edit_target(cmd: str) -> str | None:
    """Best-effort extraction of the file an edit command modifies."""
    m = _PATH_AFTER_REDIR.search(cmd)
    if m:
        return _normalise(m.group(1))
    # sed -i ... 'pattern' /path/file
    if _SED_INPLACE.search(cmd):
        m = _PATH_LAST_ARG.search(cmd)
        if m:
            return _normalise(m.group(1))
    return None


def _normalise(path: str) -> str:
    # Strip surrounding quotes; resolve trivial relative refs but don't
    # normalise across cd boundaries (we don't track cwd).
    return path.strip("'\"")


# ---- iterating tool calls across the two schemas ------------------------------


def _call_args(t: dict) -> str:
    """Pull bash command text from one tool_call entry.

    Standard schema: t['function']['arguments'] is a JSON string, with a
    'command' key inside. GPT-5-2-Codex schema: each function_call entry has
    'arguments' directly, same JSON shape.
    """
    args = (t.get("function") or {}).get("arguments") or t.get("arguments") or "{}"
    try:
        d = json.loads(args)
    except json.JSONDecodeError:
        return ""
    cmd = d.get("command") or d.get("cmd") or ""
    return " ".join(cmd) if isinstance(cmd, list) else str(cmd)


def tool_calls(trajectory: dict) -> Iterable[str]:
    """Every bash command issued, in temporal order."""
    for msg in trajectory["messages"]:
        if msg.get("role") == "assistant":
            for t in msg.get("tool_calls") or []:
                yield _call_args(t)
        elif msg.get("object") == "response" or msg.get("type") == "response":
            for o in msg.get("output") or []:
                if o.get("type") == "function_call":
                    yield _call_args(o)


# ---- per-trajectory feature record --------------------------------------------


def _is_repo_path(p: str) -> bool:
    """Heuristic: edit targets a tracked source file rather than a scratch /tmp file."""
    if not p:
        return False
    if p.startswith(("/tmp/", "/var/", "/dev/", "/root/")):
        return False
    if p in {"patch.txt", "out.txt", "result.txt"}:  # common dump targets
        return False
    return True


@dataclass
class TrajFeatures:
    instance_id: str
    n_messages: int
    n_assistant: int
    n_tool: int
    n_calls: int
    counts: dict[str, int]   # action -> count
    n_repo_edits: int        # edits targeting non-scratch paths
    n_edit_files: int        # distinct files edited
    edit_churn: float        # repo re-edits / repo edit calls
    call_unique_ratio: float
    patch_lines: int
    patch_files: int
    instance_cost: float | None
    api_calls: int | None
    exit_status: str | None


def _patch_shape(diff: str) -> tuple[int, int]:
    """(non-context line count, files touched) for a unified diff."""
    if not diff:
        return 0, 0
    lines = diff.splitlines()
    changes = sum(
        1 for ln in lines
        if (ln.startswith(("+", "-")) and not ln.startswith(("+++", "---")))
    )
    files = sum(1 for ln in lines if ln.startswith("diff --git "))
    return changes, files


def features_for(trajectory: dict, instance_id: str) -> TrajFeatures:
    msgs = trajectory["messages"]
    role_counts: dict[str, int] = {}
    for m in msgs:
        r = role_of(m)
        role_counts[r] = role_counts.get(r, 0) + 1

    cmds = list(tool_calls(trajectory))
    counts = {a: 0 for a in ACTIONS}
    edited_files: dict[str, int] = {}
    repo_edits = 0
    repo_re_edits = 0
    for cmd in cmds:
        a = classify(cmd)
        counts[a] += 1
        if a != "edit":
            continue
        tgt = edit_target(cmd)
        if not _is_repo_path(tgt or ""):
            continue
        repo_edits += 1
        if tgt in edited_files:
            repo_re_edits += 1
        edited_files[tgt] = edited_files.get(tgt, 0) + 1

    edit_churn = repo_re_edits / repo_edits if repo_edits else 0.0
    unique = len({c.strip() for c in cmds if c.strip()})
    call_unique_ratio = unique / len(cmds) if cmds else 1.0

    info = trajectory.get("info") or {}
    submission = info.get("submission") or ""
    patch_lines, patch_files = _patch_shape(submission)
    stats = info.get("model_stats") or {}

    return TrajFeatures(
        instance_id=instance_id,
        n_messages=len(msgs),
        n_assistant=role_counts.get("assistant", 0),
        n_tool=role_counts.get("tool", 0),
        n_calls=len(cmds),
        counts=counts,
        n_repo_edits=repo_edits,
        n_edit_files=len(edited_files),
        edit_churn=edit_churn,
        call_unique_ratio=call_unique_ratio,
        patch_lines=patch_lines,
        patch_files=patch_files,
        instance_cost=stats.get("instance_cost"),
        api_calls=stats.get("api_calls"),
        exit_status=info.get("exit_status"),
    )


def features_from_path(path: str | Path) -> TrajFeatures:
    p = Path(path)
    instance_id = p.name.replace(".traj.json", "")
    return features_for(json.loads(p.read_text()), instance_id)


if __name__ == "__main__":
    import sys
    f = features_from_path(sys.argv[1])
    print(json.dumps(f.__dict__, indent=2, default=str))
