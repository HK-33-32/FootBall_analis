"""Render a self-contained match viewer around one match report.

The clip becomes a data URI, which is what makes the page portable and also
what makes it far larger than the clip. For watching a match on your own
machine, `football-intelligence serve` renders the same template with the video
streamed instead -- see `football_intelligence.viewer.library`.
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_intelligence.viewer import library  # noqa: E402

TEMPLATE = library.TEMPLATE_PATH


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--template", type=Path, default=TEMPLATE)
    parser.add_argument(
        "--page-title", default="", help="browser tab and gallery name for the page"
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=None,
        help="clip to embed. It becomes a data URI, so the page stays self-contained "
        "-- keep it small enough that the whole page fits the artifact size limit.",
    )
    args = parser.parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))
    template = args.template.read_text(encoding="utf-8")
    video_uri = ""
    if args.video:
        mime = mimetypes.guess_type(args.video.name)[0] or "video/mp4"
        encoded = base64.b64encode(args.video.read_bytes()).decode("ascii")
        video_uri = f"data:{mime};base64,{encoded}"
    try:
        html = library.render(report, video_uri, template=template, title=args.page_title)
    except ValueError as exc:
        raise SystemExit(f"{args.template}: {exc}") from exc

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding="utf-8")
    size_mb = args.output.stat().st_size / 1e6
    note = f", clip {args.video.stat().st_size / 1e6:.1f} MB" if args.video else ""
    print(f"{report.get('title', args.report.stem)} -> {args.output} ({size_mb:.1f} MB{note})")


if __name__ == "__main__":
    main()
