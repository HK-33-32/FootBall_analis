"""Render a self-contained match viewer around one match report."""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
from pathlib import Path

TEMPLATE = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "football_intelligence"
    / "viewer"
    / "template.html"
)
PLACEHOLDER = "__MATCH_DATA__"


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
    if PLACEHOLDER not in template:
        raise SystemExit(f"{args.template} has no {PLACEHOLDER} placeholder")

    # The payload lands inside a <script type="application/json"> block, so the
    # only sequence that could close it early is an embedded "</script>".
    payload = json.dumps(report, ensure_ascii=False, separators=(",", ":")).replace(
        "</", "<\\/"
    )
    video_uri = ""
    if args.video:
        mime = mimetypes.guess_type(args.video.name)[0] or "video/mp4"
        encoded = base64.b64encode(args.video.read_bytes()).decode("ascii")
        video_uri = f"data:{mime};base64,{encoded}"
    html = template.replace(PLACEHOLDER, payload).replace("__VIDEO_SRC__", video_uri)
    if args.page_title:
        html = html.replace(
            "<title>Pitch Telemetry</title>", f"<title>{args.page_title}</title>", 1
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding="utf-8")
    size_mb = args.output.stat().st_size / 1e6
    note = f", clip {args.video.stat().st_size / 1e6:.1f} MB" if args.video else ""
    print(f"{report.get('title', args.report.stem)} -> {args.output} ({size_mb:.1f} MB{note})")


if __name__ == "__main__":
    main()
