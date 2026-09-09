"""Reading jersey numbers with a local vision-language model.

The fine-tuned CLIP head classifies two digits from a 224x224 crop; on this
footage it gets about half of the readable numbers right.  A general VLM reads
the shirt as text instead, and on the same tracklets it was measured at 8/9 vs
5/9 for CLIP, at ~0.26 s per crop — cheap enough because numbers are voted once
per tracklet, not per frame.

Runs through llama.cpp, so it needs a GGUF model plus its mmproj file.  If they
are not configured or not found, the caller keeps the CLIP answer.
"""
from __future__ import annotations

import base64
import collections
import os
import re

import cv2

PROMPT = ("Look at the football player in this image. If a jersey number is "
          "visible on the shirt, answer with just that number. If no number is "
          "visible or it is unreadable, answer exactly: none")

GROUP_PROMPT = ("These images all show the same football player in different "
                "frames. Reply with the jersey number on his shirt, digits "
                "only. If the number is not visible in any of them, answer "
                "exactly: none")


def _data_uri(crop) -> str:
    ok, buf = cv2.imencode(".png", crop)
    if not ok:
        raise ValueError("could not encode crop")
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()


def _parse(text: str):
    text = (text or "").strip().lower()
    if "none" in text or "no " in text:
        return None
    match = re.search(r"\d{1,2}", text)
    return int(match.group()) if match else None


class QwenJerseyReader:
    """Majority vote over the most readable crops of one tracklet."""

    def __init__(self, model_path, mmproj_path, crops_per_track=24, min_votes=1,
                 n_gpu_layers=-1, target_height=288, group_size=4):
        from llama_cpp import Llama
        from llama_cpp.llama_chat_format import Qwen25VLChatHandler

        handler = Qwen25VLChatHandler(clip_model_path=str(mmproj_path), verbose=False)
        self.llm = Llama(model_path=str(model_path), chat_handler=handler,
                         n_ctx=4096, n_gpu_layers=n_gpu_layers, verbose=False)
        self.crops_per_track = int(crops_per_track)
        self.min_votes = int(min_votes)
        # Several frames of one player go into a single question.  Measured on
        # SoccerNet ground truth: 24 crops asked four at a time gets 60% of
        # tracklets right in six queries, where asking eight crops one at a time
        # got 46% in eight.  A number is read by combining glimpses, and the
        # model can only combine what it is shown together.
        self.group_size = max(1, int(group_size))
        # every crop is scaled to the same height: small ones gain detail, and
        # close-ups do not explode into thousands of vision tokens
        self.target_height = int(target_height)

    @staticmethod
    def available(model_path, mmproj_path) -> bool:
        try:
            import llama_cpp  # noqa: F401
        except Exception:
            return False
        return bool(model_path and mmproj_path
                    and os.path.isfile(model_path) and os.path.isfile(mmproj_path))

    def _normalise(self, crop):
        height = max(1, crop.shape[0])
        scale = self.target_height / height
        if abs(scale - 1.0) <= 0.05:
            return crop
        interp = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
        return cv2.resize(crop, (max(8, int(crop.shape[1] * scale)),
                                 self.target_height), interpolation=interp)

    def _ask(self, crops) -> int | None:
        """One question about one or more views of the same player."""
        if not isinstance(crops, (list, tuple)):
            crops = [crops]
        images = [self._normalise(c) for c in crops]
        content = [{"type": "image_url", "image_url": {"url": _data_uri(c)}}
                   for c in images]
        content.append({"type": "text",
                        "text": GROUP_PROMPT if len(images) > 1 else PROMPT})
        out = self.llm.create_chat_completion(
            messages=[{"role": "user", "content": content}],
            max_tokens=8, temperature=0.0)
        return _parse(out["choices"][0]["message"]["content"])

    def _answers(self, crops):
        """One answer per group of crops, in order."""
        picked = list(crops[:self.crops_per_track])
        return [self._ask(picked[i:i + self.group_size])
                for i in range(0, len(picked), self.group_size)]

    def read(self, crops, allowed=None):
        """crops: BGR arrays, biggest first -> (number or None, answers).

        ``allowed`` (a roster's number set) is applied *before* the vote: an
        answer no squad could wear is a hallucination, and letting it into the
        count can hand the tracklet to a number that does not exist.
        """
        answers = self._answers(crops)
        valid = [a for a in answers
                 if a is not None and (allowed is None or a in allowed)]
        votes = collections.Counter(valid)
        if not votes:
            return None, answers
        number, count = votes.most_common(1)[0]
        # a single unsupported reading is a guess, not a number; with grouped
        # questions each answer already rests on several frames, so one vote
        # carries the weight that several used to
        return (number if count >= self.min_votes else None), answers

    def tally(self, crops, allowed=None):
        """Same reading, but returns every surviving answer with its count."""
        answers = self._answers(crops)
        valid = [a for a in answers
                 if a is not None and (allowed is None or a in allowed)]
        return collections.Counter(valid), answers
