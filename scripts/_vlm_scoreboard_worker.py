"""Read the broadcast score off sampled frames, inside the perception container.

Runs where the GGUF weights and llama.cpp already are, driven over ``docker
exec`` by ``scripts/read_scoreboard.py``. It answers one question per frame and
nothing else: what does the scoreboard say.
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import sys

PROMPT = (
    "This is a frame from a football broadcast. Read the score from the "
    "on-screen scoreboard graphic. Answer with two numbers separated by a "
    "dash, home first, like 2-1. If no scoreboard is visible or the score "
    "cannot be read, answer exactly: none"
)


def parse(text: str) -> tuple[int, int] | None:
    text = (text or "").strip().lower()
    if "none" in text:
        return None
    match = re.search(r"(\d{1,2})\s*[-:–]\s*(\d{1,2})", text)
    if not match:
        return None
    home, away = int(match.group(1)), int(match.group(2))
    # A broadcast score is a small number; anything else is a misread clock.
    if home > 20 or away > 20:
        return None
    return home, away


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--mmproj", required=True)
    args = parser.parse_args()

    manifest = json.load(open(args.manifest, encoding="utf-8"))
    frames = manifest["frames"]

    from llama_cpp import Llama
    from llama_cpp.llama_chat_format import Qwen25VLChatHandler

    handler = Qwen25VLChatHandler(clip_model_path=args.mmproj, verbose=False)
    llm = Llama(model_path=args.model, chat_handler=handler, n_ctx=4096, n_gpu_layers=-1,
                verbose=False)

    readings = []
    for entry in frames:
        with open(entry["path"], "rb") as handle:
            uri = "data:image/png;base64," + base64.b64encode(handle.read()).decode("ascii")
        out = llm.create_chat_completion(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": uri}},
                        {"type": "text", "text": PROMPT},
                    ],
                }
            ],
            max_tokens=8,
            temperature=0.0,
        )
        answer = out["choices"][0]["message"]["content"]
        score = parse(answer)
        readings.append(
            {
                "frame": entry["frame"],
                "home": None if score is None else score[0],
                "away": None if score is None else score[1],
                "answer": answer.strip(),
            }
        )
        print(f"frame {entry['frame']}: {answer.strip()}", flush=True)

    json.dump(
        {"readings": readings}, open(args.output, "w", encoding="utf-8"), ensure_ascii=False,
        indent=1,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
