"""Ask whether a referee is showing a card, inside the perception container.

Driven over ``docker exec`` by ``scripts/read_cards.py``. One question per
frame, and a deliberately narrow one: models are eager to see a card in any
raised arm, so the prompt asks for the card itself and offers "none" as the
expected answer.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys

# Asked loosely ("is the referee showing a card?") the model says yes to almost
# any frame: measured on three frames of open play with no card in them, two
# came back "yellow". The question has to describe the thing itself -- a small
# rectangle held above the head, arm extended upward -- and make "none" the
# expected answer rather than the exception.
PROMPT = (
    "Look at this football broadcast frame. A referee showing a card holds a "
    "small rectangular card in a raised hand, arm extended above shoulder "
    "height, clearly separated against the background. Almost always there is "
    "no card in a frame; open play, running players and a referee with arms "
    "down all mean there is no card. Answer with exactly one word: none, "
    "yellow, or red. Answer none unless you can actually see the rectangular "
    "card held up in the air."
)


def parse(text: str) -> str | None:
    text = (text or "").strip().lower()
    if "yellow" in text:
        return "yellow"
    if "red" in text:
        return "red"
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--mmproj", required=True)
    args = parser.parse_args()

    manifest = json.load(open(args.manifest, encoding="utf-8"))

    from llama_cpp import Llama
    from llama_cpp.llama_chat_format import Qwen25VLChatHandler

    handler = Qwen25VLChatHandler(clip_model_path=args.mmproj, verbose=False)
    llm = Llama(model_path=args.model, chat_handler=handler, n_ctx=4096, n_gpu_layers=-1,
                verbose=False)

    readings = []
    for entry in manifest["frames"]:
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
            max_tokens=4,
            temperature=0.0,
        )
        answer = out["choices"][0]["message"]["content"]
        readings.append(
            {"frame": entry["frame"], "colour": parse(answer), "answer": answer.strip()}
        )
        print(f"frame {entry['frame']}: {answer.strip()}", flush=True)

    json.dump(
        {"readings": readings}, open(args.output, "w", encoding="utf-8"),
        ensure_ascii=False, indent=1,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
