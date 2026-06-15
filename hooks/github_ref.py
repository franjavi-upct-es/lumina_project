"""MkDocs hook: expand a `[gh:...]` shorthand into a GitHub code-reference badge.

Syntax in any Markdown page:

    [gh:backend/cognition/training/behavioral_cloning.py#L20-L25]
    [gh:backend/cognition/training/behavioral_cloning.py#L20]
    [gh:backend/cognition/training/behavioral_cloning.py]

Each is rewritten to a styled link (class `gh-ref`, see assets/css/github-ref.css)
pointing at the file on GitHub. The repo URL is taken from `repo_url` in mkdocs.yml,
and the branch from `extra.repo_branch` (defaults to "main").
"""

import re

# [gh:<path>] or [gh:<path>#<anchor>]
_PATTERN = re.compile(r"\[gh:([^\]#]+)(?:#([^\]]+))?\]")


def on_page_markdown(markdown, page, config, files):
    repo_url = (config.get("repo_url") or "").rstrip("/")
    branch = (config.get("extra") or {}).get("repo_branch", "main")

    def replace(match):
        path = match.group(1).strip()
        anchor = (match.group(2) or "").strip()  # e.g. "L20-L25"

        url = f"{repo_url}/blob/{branch}/{path}"
        # Markdown files need ?plain=1 for line anchors to resolve on GitHub.
        if path.endswith(".md"):
            url += "?plain=1"
        if anchor:
            url += f"#{anchor}"
            # "L20-L25" -> "20-25", "L20" -> "20" for display
            display_lines = re.sub(r"L(\d+)", r"\1", anchor)
            label = f":material-github: {path}<span class=\"lines\">#{display_lines}</span>"
        else:
            label = f":material-github: {path}"

        return f"[{label}]({url}){{ .gh-ref }}"

    return _PATTERN.sub(replace, markdown)
