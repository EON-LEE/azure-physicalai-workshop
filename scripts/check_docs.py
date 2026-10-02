"""Check repository-local Markdown file links without network or cloud access."""

from __future__ import annotations

import json
import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

LINK = re.compile(r"!?\[[^\]\n]*\]\(\s*(<[^>\n]+>|[^\s)]+)(?:\s+[^)\n]*)?\)")
REFERENCE = re.compile(r"^\s{0,3}\[[^\]\n]+\]:\s*(<[^>\n]+>|\S+)", re.MULTILINE)


def markdown_targets(text: str):
    fence = None
    lines = []
    for line in text.splitlines():
        marker = re.match(r"^\s{0,3}(`{3,}|~{3,})", line)
        if marker:
            current = marker[1]
            if fence is None:
                fence = current
            elif current[0] == fence[0] and len(current) >= len(fence):
                fence = None
            continue
        if fence is None:
            lines.append(line)
    content = "\n".join(lines)
    for pattern in (LINK, REFERENCE):
        for match in pattern.finditer(content):
            yield match[1].strip("<>")


def check_links(root: Path, documents: list[Path]) -> list[dict[str, str]]:
    root = root.resolve()
    issues = []
    for document in documents:
        for target in markdown_targets(document.read_text(encoding="utf-8")):
            parsed = urlsplit(target)
            if parsed.scheme or parsed.netloc or not parsed.path:
                continue
            path = unquote(parsed.path).replace("\\", "/")
            resolved = (document.parent / path).resolve()
            if not resolved.is_relative_to(root):
                reason = "outside_repository"
            elif not resolved.exists():
                reason = "missing_target"
            else:
                continue
            issues.append(
                {
                    "document": document.relative_to(root).as_posix(),
                    "target": target,
                    "reason": reason,
                }
            )
    return issues


def documents(root: Path) -> list[Path]:
    return sorted(
        [*root.glob("*.md"), *(root / "docs").rglob("*.md")],
        key=lambda path: path.relative_to(root).as_posix(),
    )


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    paths = documents(root)
    issues = check_links(root, paths)
    print(
        json.dumps(
            {
                "valid": not issues,
                "documents": len(paths),
                "scope": "local_markdown_file_targets",
                "anchors_checked": False,
                "remote_urls_checked": False,
                "issues": issues,
            },
            indent=2,
        )
    )
    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
