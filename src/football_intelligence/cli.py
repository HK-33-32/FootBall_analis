from __future__ import annotations

import argparse
import json
from pathlib import Path

from .api.app import create_app
from .evaluation import semantic_metrics
from .memory import MatchMemory
from .pipeline import MatchPipeline


def main() -> None:
    parser = argparse.ArgumentParser(prog="football-intelligence")
    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8080)
    serve.add_argument("--data-dir", default="./data/runtime")
    evaluate = commands.add_parser("evaluate-semantic")
    evaluate.add_argument("annotations", help="JSONL rows with truth, prediction, confidence")
    evaluate.add_argument("--output", default="")
    import_legacy = commands.add_parser("import-legacy")
    import_legacy.add_argument("--match-id", required=True)
    import_legacy.add_argument("--video", required=True)
    import_legacy.add_argument("--events", required=True)
    import_legacy.add_argument("--predictions", required=True)
    import_legacy.add_argument("--start", type=float, default=0)
    import_legacy.add_argument("--end", type=float, required=True)
    import_legacy.add_argument("--fps", type=int, default=12)
    import_legacy.add_argument("--data-dir", default="./data/runtime")
    import_legacy.add_argument("--output", default="")
    args = parser.parse_args()
    if args.command == "serve":
        import uvicorn

        uvicorn.run(create_app(args.data_dir), host=args.host, port=args.port)
    elif args.command == "evaluate-semantic":
        rows = [
            json.loads(line)
            for line in Path(args.annotations).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        result = semantic_metrics(rows)
        text = json.dumps(result, ensure_ascii=False, indent=2)
        if args.output:
            Path(args.output).write_text(text, encoding="utf-8")
        print(text)
    elif args.command == "import-legacy":
        memory = MatchMemory(Path(args.data_dir) / "match_memory.sqlite")
        pipeline = MatchPipeline(memory=memory, semantic_engine=None)
        result = pipeline.run(
            match_id=args.match_id,
            video_path=args.video,
            start_s=args.start,
            end_s=args.end,
            fps=args.fps,
            legacy_events=json.loads(Path(args.events).read_text(encoding="utf-8")),
            predictions=json.loads(Path(args.predictions).read_text(encoding="utf-8")),
            run_semantics=False,
            dataset="legacy-development-artifact",
            split="development",
        )
        text = json.dumps(result, ensure_ascii=False, indent=2)
        if args.output:
            Path(args.output).write_text(text, encoding="utf-8")
        print(
            json.dumps(
                {
                    "match_id": result["match_id"],
                    "tracklets": result["tracklets"],
                    "global_players": len(result["global_players"]),
                    "candidates": len(result["candidates"]),
                    "output": args.output or "not written",
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
