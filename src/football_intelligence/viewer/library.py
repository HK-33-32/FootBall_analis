"""Serving finished match reports instead of shipping each one as a file.

``build_match_viewer.py`` bakes a report and its clip into a single page: the
clip becomes a data URI, which is what makes the page portable and also what
makes it thirty times larger than the clip. That trade is right for something
you hand to someone else and wrong for something you watch on your own machine,
where the video should stream and seeking should work.

This module is the other half. It finds the reports on a mounted directory,
renders the same template around one of them, and points the page at a URL the
server will stream. The page is otherwise identical, so there is one viewer to
maintain rather than two.

Report ids are directory names, and they are validated rather than trusted: the
id arrives from a URL and is used to build a path, so anything but a plain name
is refused before it reaches the filesystem.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPORT_NAME = "match_report.json"
TEMPLATE_PATH = Path(__file__).with_name("template.html")
DATA_PLACEHOLDER = "__MATCH_DATA__"
VIDEO_PLACEHOLDER = "__VIDEO_SRC__"
TITLE_TAG = "<title>Pitch Telemetry</title>"

# A clip built for the browser is preferred; the annotated comparison renders
# are diagnostics, not the match, and never belong behind the telemetry.
PREFERRED_VIDEOS = ("clip_web.mp4", "clip.mp4", "video.mp4", "match.mp4")
VIDEO_SUFFIXES = (".mp4", ".webm", ".m4v")
DIAGNOSTIC_PREFIXES = ("annotated",)

SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


@dataclass(frozen=True)
class Report:
    """One finished match sitting on disk."""

    id: str
    title: str
    path: Path
    video: Path | None
    frames: int
    covered: int
    players: int
    modified: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "frames": self.frames,
            "covered": self.covered,
            "players": self.players,
            "has_video": self.video is not None,
            "modified": round(self.modified, 3),
        }


def _pick_video(directory: Path) -> Path | None:
    for name in PREFERRED_VIDEOS:
        candidate = directory / name
        if candidate.is_file():
            return candidate
    others = sorted(
        item
        for item in directory.iterdir()
        if item.is_file()
        and item.suffix.lower() in VIDEO_SUFFIXES
        and not item.name.lower().startswith(DIAGNOSTIC_PREFIXES)
    )
    return others[0] if others else None


def _summarise(path: Path, report_id: str) -> Report | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    players = payload.get("players")
    return Report(
        id=report_id,
        title=str(payload.get("title") or report_id),
        path=path,
        video=_pick_video(path.parent),
        frames=int(payload.get("clip_frames") or 0),
        covered=int(payload.get("frames_with_game_state") or 0),
        players=len(players) if isinstance(players, list | dict) else 0,
        modified=path.stat().st_mtime,
    )


def discover(root: Path | str) -> list[Report]:
    """Every readable report under ``root``, newest first.

    A directory holds one match: the report, and beside it the clip it came
    from. Directories that hold something else, or a report that will not
    parse, are skipped rather than failing the listing -- a half-written run
    should not take the library down with it.
    """
    root = Path(root)
    if not root.is_dir():
        return []
    found: list[Report] = []
    for path in sorted(root.glob(f"*/{REPORT_NAME}")):
        report_id = path.parent.name
        if not SAFE_ID.match(report_id):
            continue
        summary = _summarise(path, report_id)
        if summary is not None:
            found.append(summary)
    return sorted(found, key=lambda item: -item.modified)


def find(root: Path | str, report_id: str) -> Report | None:
    """One report by id, or None. The id is never trusted as a path."""
    if not SAFE_ID.match(report_id or ""):
        return None
    path = Path(root) / report_id / REPORT_NAME
    if not path.is_file():
        return None
    return _summarise(path, report_id)


def render(
    report: dict[str, Any],
    video_src: str = "",
    template: str | None = None,
    title: str = "",
) -> str:
    """Put one report inside the viewer template.

    ``video_src`` is either a URL the server will stream or a data URI for a
    page that has to stand on its own.
    """
    template = template if template is not None else TEMPLATE_PATH.read_text(encoding="utf-8")
    if DATA_PLACEHOLDER not in template:
        raise ValueError(f"the template has no {DATA_PLACEHOLDER} placeholder")
    # The payload lands inside a <script type="application/json"> block, so the
    # only sequence that could close it early is an embedded "</script>".
    payload = json.dumps(report, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    html = template.replace(DATA_PLACEHOLDER, payload).replace(VIDEO_PLACEHOLDER, video_src)
    if title:
        html = html.replace(TITLE_TAG, f"<title>{_escape(title)}</title>", 1)
    return html


def _escape(text: str) -> str:
    return (
        text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
    )


__all__ = ["Report", "discover", "find", "render", "TEMPLATE_PATH"]
