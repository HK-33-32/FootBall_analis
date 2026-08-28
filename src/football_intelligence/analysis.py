"""Running a whole match end to end, with something honest to show while it runs.

The pieces already existed as scripts. What was missing is the thing a person
needs when a job takes hours: knowing where it is and when it will finish.

The estimate is not a guess at a fixed rate. Screening measures how much of the
footage is actually football, and that -- not the length of the file -- is what
costs. Then every finished chunk reports how long its seconds of football took,
and the remaining time is re-derived from the rate this machine is actually
achieving on this video. The first estimate uses a measured default of 24x real
time; by the second chunk the number is the machine's own.

Progress is weighted the same way the work is: screening is a rounding error,
perception is nearly all of it, and the analysis at the end is seconds.

A run that takes hours will sooner or later meet something that stops it: a
machine that reboots, a container that dies, a power cut. Every finished chunk
is therefore written to disk beside the report, along with the plan that
produced it, and a run that starts against a directory that already holds them
picks up where it stopped. The cost of an interruption is one chunk, not the
day.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .calibration import CalibrationConfig, drop_uncalibrated
from .gamestate import RefinementConfig, VideoFrames, refine_predictions
from .longmatch import ChunkConfig, merge_chunks, plan_chunks
from .match_stats import StatsConfig, match_statistics
from .playability import PlayabilityConfig, screen
from .trajectories import TrajectoryConfig

# Measured against the football-core backend after its per-frame model calls
# were batched: 7.6 s of football in 180 s, and a 30 s clip in 427 s.
DEFAULT_RATE = 24.0

# What each stage is worth of the whole, by how long it actually takes.
WEIGHTS = {"screen": 0.02, "perception": 0.92, "refine": 0.04, "report": 0.02}

STAGE_TITLES = {
    "queued": "В очереди",
    "screen": "Отбор пригодных фрагментов",
    "perception": "Детекция, трекинг, номера",
    "refine": "Уточнение игрового состояния",
    "report": "Статистика и события",
    "done": "Готово",
    "error": "Ошибка",
    "canceled": "Отменено",
}


@dataclass
class AnalysisConfig:
    """Where the pieces live and how they talk to each other."""

    core_url: str = "http://host.docker.internal:8000"
    # Where this process writes clips for the perception backend...
    media_dir: Path = Path("/data/perception_benchmark/media")
    # ...and how that same directory looks from inside the backend.
    media_container_root: str = "/data/media"
    reports_dir: Path = Path("/reports")
    fps: int = 25
    chunk_s: float = 60.0
    min_grass_share: float = 0.40
    poll_s: float = 3.0
    # A pause between chunks. Zero by default; raise it on a machine that grows
    # unstable under hours of unbroken all-core load.
    cooldown_s: float = 0.0


@dataclass
class Progress:
    """Everything the page needs to draw a bar and a number of minutes."""

    stage: str = "queued"
    fraction: float = 0.0
    message: str = ""
    playable_s: float = 0.0
    duration_s: float = 0.0
    chunks_total: int = 0
    chunks_done: int = 0
    # Seconds of football already analysed, including the chunk in flight. A
    # chunk is a minute long, so counting only finished ones leaves the
    # estimate frozen for twenty minutes at a time.
    analysed_s: float = 0.0
    rate: float = DEFAULT_RATE
    rate_measured: bool = False
    started_at: float = field(default_factory=time.time)
    eta_s: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "stage_title": STAGE_TITLES.get(self.stage, self.stage),
            "fraction": round(min(max(self.fraction, 0.0), 1.0), 4),
            "message": self.message,
            "playable_s": round(self.playable_s, 1),
            "duration_s": round(self.duration_s, 1),
            "chunks_total": self.chunks_total,
            "chunks_done": self.chunks_done,
            "analysed_s": round(self.analysed_s, 1),
            "rate": round(self.rate, 1),
            "rate_measured": self.rate_measured,
            "elapsed_s": round(time.time() - self.started_at, 1),
            "eta_s": None if self.eta_s is None else round(self.eta_s),
        }


def estimate_remaining(progress: Progress) -> float:
    """Seconds left, from the football still to analyse at the current rate."""
    if progress.stage in ("done", "error", "canceled"):
        return 0.0
    if progress.stage == "queued" or progress.chunks_total == 0:
        # Before screening, the only thing known is the length of the file, and
        # roughly half of a broadcast is not football.
        return progress.duration_s * 0.58 * progress.rate if progress.duration_s else 0.0
    remaining_football = max(0.0, progress.playable_s - progress.analysed_s)
    tail = 6.0 if progress.stage in ("screen", "perception") else 2.0
    return remaining_football * progress.rate + tail


def video_duration_s(path: Path) -> float:
    """Length of the file, read from its header rather than by decoding it."""
    import cv2

    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            return 0.0
        fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
        frames = capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
        return float(frames) / float(fps) if fps else 0.0
    finally:
        capture.release()


def transcode(source: Path, target: Path, start: float, duration: float, fps: int) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-ss", f"{start:.3f}", "-i", str(source), "-t", f"{duration:.3f}",
            "-an", "-vf", f"scale=1920:1080:flags=lanczos,fps={fps}",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
            str(target),
        ],
        check=True,
    )


def web_clip(source: Path, target: Path) -> None:
    """A small copy for the viewer to stream behind the telemetry."""
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error", "-i", str(source),
            "-an", "-vf", "scale=960:-2", "-c:v", "libx264", "-preset", "veryfast",
            "-crf", "30", "-movflags", "+faststart", str(target),
        ],
        check=True,
    )


class CoreClient:
    """The perception backend, over HTTP."""

    def __init__(self, base_url: str, poll_s: float = 3.0):
        self.base_url = base_url.rstrip("/")
        self.poll_s = poll_s

    def _client(self):
        import httpx

        return httpx.Client(timeout=60.0)

    def healthy(self) -> bool:
        try:
            with self._client() as client:
                return client.get(f"{self.base_url}/v1/videos").status_code == 200
        except Exception:
            return False

    def analyse(
        self,
        media_path: str,
        name: str,
        duration: float,
        fps: int,
        on_progress: Callable[[float], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> list[dict[str, Any]]:
        import httpx

        with self._client() as client:
            video = client.post(
                f"{self.base_url}/v1/videos/register", data={"path": media_path, "name": name}
            )
            video.raise_for_status()
            job = client.post(
                f"{self.base_url}/v1/jobs",
                json={
                    "video_id": video.json()["id"], "start": 0, "end": duration, "fps": fps,
                    "resolution": "1080", "detector": "rfdetr", "detector_size": "large",
                    "jersey_mode": "CLIP", "jersey_reader": "qwen-vl", "jersey_stride": 3,
                    "calib_stride": 1, "render_video": False, "analytics": False,
                },
            )
            job.raise_for_status()
            job_id = job.json()["id"]
            while True:
                if should_stop is not None and should_stop():
                    client.post(f"{self.base_url}/v1/jobs/{job_id}/cancel")
                    raise RuntimeError("отменено пользователем")
                state = client.get(f"{self.base_url}/v1/jobs/{job_id}").json()
                if on_progress is not None:
                    on_progress(float(state.get("progress") or 0.0))
                if state["status"] == "done":
                    break
                if state["status"] in ("error", "canceled"):
                    raise RuntimeError(state.get("error") or state["status"])
                time.sleep(self.poll_s)
            with httpx.stream(
                "GET", f"{self.base_url}/v1/jobs/{job_id}/predictions", timeout=None
            ) as response:
                response.raise_for_status()
                body = b"".join(response.iter_bytes(1024 * 1024))
        return json.loads(body)["predictions"]


class Analysis:
    """One match being analysed, start to finish."""

    def __init__(
        self,
        video: Path,
        title: str,
        config: AnalysisConfig,
        roster: dict[str, Any] | None = None,
        analysis_id: str | None = None,
        resume: bool = True,
    ):
        self.id = analysis_id or uuid.uuid4().hex[:12]
        self.resume = resume
        self.video = Path(video)
        self.title = title or self.video.stem
        self.config = config
        self.roster = roster
        self.progress = Progress()
        self.error: str | None = None
        self.report_path: Path | None = None
        self._stop = threading.Event()

    # ------------------------------------------------------------------ state
    def as_dict(self) -> dict[str, Any]:
        payload = {
            "id": self.id,
            "title": self.title,
            "video": self.video.name,
            "error": self.error,
            "report": str(self.report_path) if self.report_path else None,
            **self.progress.as_dict(),
        }
        payload["eta_s"] = (
            None if self.progress.stage in ("done", "error", "canceled")
            else round(estimate_remaining(self.progress))
        )
        return payload

    def cancel(self) -> None:
        self._stop.set()

    def _step(self, stage: str, within: float, message: str = "") -> None:
        base = 0.0
        for name, weight in WEIGHTS.items():
            if name == stage:
                break
            base += weight
        self.progress.stage = stage
        self.progress.fraction = base + WEIGHTS.get(stage, 0.0) * min(max(within, 0.0), 1.0)
        if message:
            self.progress.message = message

    # -------------------------------------------------------------------- run
    def run(self) -> None:
        try:
            self._run()
        except Exception as exc:  # the page has to say what went wrong
            self.error = f"{type(exc).__name__}: {exc}"
            self.progress.stage = "canceled" if self._stop.is_set() else "error"
            self.progress.message = self.error

    def _run(self) -> None:
        target = Path(self.config.reports_dir) / self.id
        target.mkdir(parents=True, exist_ok=True)
        chunk_dir = target / "chunks"
        chunk_dir.mkdir(exist_ok=True)
        plan_file = target / "plan.json"

        # Knowing the length before screening is what lets the first estimate
        # be a number rather than a shrug.
        self.progress.duration_s = video_duration_s(self.video)
        plan = None
        if self.resume and plan_file.is_file():
            try:
                plan = json.loads(plan_file.read_text(encoding="utf-8"))
            except ValueError:
                plan = None
        if plan is not None:
            scan, chunks = plan["screen"], plan["chunks"]
            self._step("screen", 1.0, "Продолжаю прерванный расчёт")
        else:
            self._step("screen", 0.1, "Ищу фрагменты, на которых видно поле")
            scan = screen(
                str(self.video), PlayabilityConfig(min_grass_share=self.config.min_grass_share)
            )
            chunks = plan_chunks(
                scan["segments"], ChunkConfig(max_chunk_s=self.config.chunk_s, fps=self.config.fps)
            )
            plan_file.write_text(
                json.dumps(
                    {"screen": scan, "chunks": chunks, "video": str(self.video),
                     "title": self.title},
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
        self.progress.duration_s = float(scan["duration_s"])
        self.progress.playable_s = float(scan["playable_s"])
        self.progress.chunks_total = len(chunks)
        if not chunks:
            raise RuntimeError("на записи не нашлось ни одного игрового фрагмента")
        already = len(list(chunk_dir.glob("*.json"))) if self.resume else 0
        self._step(
            "screen", 1.0,
            f"{scan['playable_s']:.0f} с футбола из {scan['duration_s']:.0f} с "
            f"({scan['playable_share']:.0%}), {len(chunks)} кусков"
            + (f"; {already} уже посчитано ранее" if already else ""),
        )

        core = CoreClient(self.config.core_url, self.config.poll_s)
        if not core.healthy():
            raise RuntimeError(f"контейнер перцепции недоступен: {self.config.core_url}")

        parts: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
        analysed_s = 0.0
        skipped_s = 0.0
        perception_started = time.time()
        for index, chunk in enumerate(chunks, start=1):
            if self._stop.is_set():
                raise RuntimeError("отменено пользователем")
            done_file = chunk_dir / f"{chunk['name']}.json"
            if self.resume and done_file.is_file():
                try:
                    saved = json.loads(done_file.read_text(encoding="utf-8"))["predictions"]
                except (ValueError, KeyError):
                    saved = None
                if saved is not None:
                    parts.append((chunk, saved))
                    analysed_s += chunk["duration_s"]
                    self.progress.analysed_s = analysed_s
                    self.progress.chunks_done = index
                    self._step(
                        "perception",
                        analysed_s / max(self.progress.playable_s, 0.001),
                        f"Кусок {index} из {len(chunks)} уже посчитан — пропускаю",
                    )
                    # restored in no time, so it is not evidence of pace
                    skipped_s += chunk["duration_s"]
                    continue
            name = f"{self.id}-{chunk['name']}"
            media = Path(self.config.media_dir) / name / f"{name}.mp4"
            transcode(self.video, media, chunk["start_s"], chunk["duration_s"], self.config.fps)

            def on_chunk(value: float, index=index, chunk=chunk, analysed_s=analysed_s) -> None:
                # The backend runs its stages one after another and restarts
                # its own bar for each, so its progress goes backwards. A bar
                # that retreats is worse than a coarse one: keep the best seen.
                done = analysed_s + value * chunk["duration_s"]
                self.progress.analysed_s = max(self.progress.analysed_s, done)
                inside = self.progress.analysed_s / max(self.progress.playable_s, 0.001)
                self._step(
                    "perception", inside,
                    f"Кусок {index} из {len(chunks)} — {chunk['start_s']:.0f}–"
                    f"{chunk['end_s']:.0f} с записи",
                )

            predictions = core.analyse(
                f"{self.config.media_container_root}/{name}/{name}.mp4",
                name,
                chunk["duration_s"],
                self.config.fps,
                on_progress=on_chunk,
                should_stop=self._stop.is_set,
            )
            # written before anything else so an interruption costs this
            # chunk at most, not the hours already spent
            done_file.write_text(
                json.dumps({"predictions": predictions}, ensure_ascii=False), encoding="utf-8"
            )
            parts.append((chunk, predictions))
            analysed_s += chunk["duration_s"]
            self.progress.analysed_s = analysed_s
            self.progress.chunks_done = index
            # the rate this machine is actually managing, over the football it
            # actually analysed -- chunks restored from disk are not evidence
            worked_s = analysed_s - skipped_s
            if worked_s > 0:
                self.progress.rate = (time.time() - perception_started) / worked_s
                self.progress.rate_measured = True
            if self.config.cooldown_s and index < len(chunks):
                self._step(
                    "perception",
                    analysed_s / max(self.progress.playable_s, 0.001),
                    f"Пауза {self.config.cooldown_s:.0f} с между кусками",
                )
                self._stop.wait(self.config.cooldown_s)

        merged = merge_chunks(parts, ChunkConfig(fps=self.config.fps))
        (target / "predictions.json").write_text(
            json.dumps({"predictions": merged["predictions"]}, ensure_ascii=False),
            encoding="utf-8",
        )

        self._step("refine", 0.2, "Команды, роли и номера по всей записи")
        full_clip = Path(self.config.media_dir) / self.id / f"{self.id}.mp4"
        if not full_clip.is_file():
            transcode(self.video, full_clip, 0.0, scan["duration_s"], self.config.fps)
        refined, refinement = refine_predictions(
            merged["predictions"],
            VideoFrames(str(full_clip)),
            RefinementConfig(link_tracklets=True),
        )
        (target / "predictions_refined.json").write_text(
            json.dumps({"predictions": refined}, ensure_ascii=False), encoding="utf-8"
        )
        (target / "refinement.json").write_text(
            json.dumps(refinement, ensure_ascii=False, indent=1), encoding="utf-8"
        )

        self._step("report", 0.3, "Считаю статистику и события")
        kept, calibration = drop_uncalibrated(refined, CalibrationConfig())
        report = match_statistics(
            kept, StatsConfig(fps=self.config.fps), TrajectoryConfig(fps=self.config.fps)
        )
        report["title"] = self.title
        report["source"] = str(target / "predictions_refined.json")
        report["calibration"] = {k: v for k, v in calibration.items() if k != "rejected"}
        if self.roster:
            report["roster"] = self.roster
        (target / "match_report.json").write_text(
            json.dumps(report, ensure_ascii=False), encoding="utf-8"
        )
        self.report_path = target / "match_report.json"

        self._step("report", 0.8, "Готовлю запись для витрины")
        try:
            web_clip(full_clip, target / "clip_web.mp4")
        except (subprocess.CalledProcessError, OSError):
            shutil.copy2(full_clip, target / "clip_web.mp4")

        self.progress.stage = "done"
        self.progress.fraction = 1.0
        self.progress.eta_s = 0
        self.progress.message = (
            f"{len(report.get('players', []))} игроков, "
            f"{report.get('frames_with_game_state', 0)} кадров с состоянием"
        )


class AnalysisManager:
    """The analyses this process is running, and the ones it has finished."""

    def __init__(self, config: AnalysisConfig, workers: int = 1):
        self.config = config
        self._jobs: dict[str, Analysis] = {}
        self._lock = threading.Lock()
        self._queue: list[str] = []
        self._workers = workers
        self._busy = 0

    def submit(
        self,
        video: Path,
        title: str,
        roster: dict[str, Any] | None = None,
        analysis_id: str | None = None,
    ) -> Analysis:
        analysis = Analysis(video, title, self.config, roster, analysis_id=analysis_id)
        with self._lock:
            self._jobs[analysis.id] = analysis
            self._queue.append(analysis.id)
        self._pump()
        return analysis

    def _pump(self) -> None:
        with self._lock:
            if self._busy >= self._workers or not self._queue:
                return
            analysis = self._jobs[self._queue.pop(0)]
            self._busy += 1

        def run() -> None:
            try:
                analysis.run()
            finally:
                with self._lock:
                    self._busy -= 1
                self._pump()

        threading.Thread(target=run, name=f"analysis-{analysis.id}", daemon=True).start()

    def get(self, analysis_id: str) -> Analysis | None:
        return self._jobs.get(analysis_id)

    def interrupted(self) -> list[dict[str, Any]]:
        """Runs on disk that were planned but never finished.

        The manager keeps its state in memory, so a machine that reboots takes
        the list of running jobs with it. What survives is the work: the plan
        and the chunks already analysed. This finds those, so an interrupted
        match can be picked up rather than started again.
        """
        found: list[dict[str, Any]] = []
        root = Path(self.config.reports_dir)
        if not root.is_dir():
            return found
        for plan_file in sorted(root.glob("*/plan.json")):
            directory = plan_file.parent
            if (directory / "match_report.json").is_file():
                continue
            if directory.name in self._jobs and self._jobs[directory.name].progress.stage not in (
                "error",
                "canceled",
            ):
                continue
            try:
                plan = json.loads(plan_file.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            chunks = plan.get("chunks") or []
            done = len(list((directory / "chunks").glob("*.json")))
            found.append(
                {
                    "id": directory.name,
                    "title": plan.get("title") or directory.name,
                    "video": plan.get("video"),
                    "chunks_total": len(chunks),
                    "chunks_done": done,
                    "playable_s": round(float((plan.get("screen") or {}).get("playable_s", 0)), 1),
                }
            )
        return found

    def resume(self, analysis_id: str) -> Analysis | None:
        """Start an interrupted run again; the chunks on disk are not redone."""
        plan_file = Path(self.config.reports_dir) / analysis_id / "plan.json"
        if not plan_file.is_file():
            return None
        plan = json.loads(plan_file.read_text(encoding="utf-8"))
        video = Path(plan.get("video", ""))
        if not video.is_file():
            raise FileNotFoundError(str(video))
        return self.submit(video, plan.get("title", analysis_id), analysis_id=analysis_id)

    def list(self) -> list[dict[str, Any]]:
        return [
            job.as_dict()
            for job in sorted(self._jobs.values(), key=lambda item: -item.progress.started_at)
        ]


__all__ = [
    "Analysis",
    "AnalysisConfig",
    "AnalysisManager",
    "CoreClient",
    "Progress",
    "estimate_remaining",
]
