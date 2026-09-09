"""Runs the SoccerNet GSR pipeline for one clip and reports progress.

Stage layout (the two heavy stages are independent, so they run side by side):

    extract ──► ┌ calibration (sharded over processes, CPU bound) ┐ ──► IDATR
                └ detection + tracking + jersey (GPU bound)       ┘      │
                                                                        ▼
                                                     visualize ──► encode
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import yaml

from .settings import (CALIB_WORKERS_MAX, CHECKPOINT_DIR, RAM_PER_WORKER_GB, RAM_RESERVE_GB,
                       SEGMENT_SECONDS, VIS_WORKERS)
from .paths import JOB_DIR, REPO, ROOT, RUNNERS, VENV_PYTHON, free_ram_gb
from .segments import concat_videos, merge_predictions, plan_segments

TQDM_RE = re.compile(r"(\d+)\s*/\s*(\d+)")
# tqdm bars of one-off model downloads must not be read as pipeline progress
DOWNLOAD_RE = re.compile(r"i?B/s")
MAX_LOG_LINES = 600

# Each calibration worker is a separate CUDA process (~1.6 GB of host RAM
# measured) and the tracking process runs alongside them, so the worker count
# is capped by the free RAM at job start.


# The tracking process runs alongside and grows *after* this measurement is
# taken (RF-DETR compile + CLIP ViT-L + ReID peak around 6 GB), so the reserve
# has to cover it plus room for the OS.


# Long intervals are processed segment by segment: frames are extracted,
# analysed, rendered and deleted before the next segment starts, so disk and
# memory stay flat regardless of how long the match is.



def calib_workers(job=None) -> int:
    if job is not None and getattr(job, "calib_workers_override", None):
        return int(job.calib_workers_override)
    free = free_ram_gb()
    if not free:
        return 2
    return max(1, min(CALIB_WORKERS_MAX, int((free - RAM_RESERVE_GB) // RAM_PER_WORKER_GB)))


@dataclass
class Part:
    """One process inside a stage."""
    label: str
    build: Callable[["Job"], list]
    total: Optional[Callable[["Job"], int]] = None
    count: Optional[Callable[["Job"], int]] = None


@dataclass
class Stage:
    key: str
    title: str
    weight: float
    parts: list = field(default_factory=list)


class Job:
    def __init__(self, *, video: dict, start: float, end: float, fps: int,
                 jersey_mode: str, resolution: str, detector: str = "rfdetr",
                 detector_size: str = "medium", jersey_stride: int = 2,
                 calib_stride: int = 1, jersey_reader: str = "clip",
                 segment_seconds: float = SEGMENT_SECONDS, roster: str = "",
                 render_video: bool = True, render_style: str = "fifa",
                 analytics: bool = True,
                 job_id: Optional[str] = None, clip_name: Optional[str] = None):
        self.id = job_id or uuid.uuid4().hex[:12]
        self.video = video
        self.start = float(start)
        self.end = float(end)
        self.fps = int(fps)
        self.jersey_mode = jersey_mode
        self.resolution = resolution
        self.detector = detector
        self.detector_size = detector_size
        self.jersey_stride = int(jersey_stride)
        self.calib_stride = int(calib_stride)
        self.jersey_reader = jersey_reader
        self.segment_seconds = float(segment_seconds)
        self.roster = roster or ""
        # Drawing every frame and re-encoding costs about a sixth of the run and
        # tens of gigabytes; a statistics run needs neither.  Off by choice, not
        # by removal — flip it back and the annotated video returns.
        self.render_video = bool(render_video)
        self.render_style = render_style if render_style in ("fifa", "boxes") else "fifa"
        self.analytics = bool(analytics)
        self.segments = plan_segments(self.start, self.end, self.segment_seconds)
        self.segment_index = 0
        self.calib_workers_override = None

        self.clip_name = clip_name or "SNGS-%05d" % (int(time.time()) % 100000)
        self.dir = JOB_DIR / self.id
        self.result_path = self.dir / "result.mp4"
        self.predictions_path = self.dir / "predictions.json"
        self._use_segment(0)

        self.status = "queued"
        self.stage_key = ""
        self.stage_title = ""
        self.stage_progress = 0.0
        self.progress = 0.0
        self.error = ""
        self.log: list[str] = []
        self.created_at = time.time()
        self.started_at: Optional[float] = None
        self.finished_at: Optional[float] = None
        self.timings: dict[str, float] = {}
        self.expected_frames = max(1, int(round((self.end - self.start) * self.fps)))

        # set by the API layer; carried through to the callback and the ledger
        self.metadata: dict = {}
        self.kickoff_offset_s = None
        self.period = None

        self._procs: list[subprocess.Popen] = []
        self._cancel = False
        self._lock = threading.Lock()

    # ------------------------------------------------------------- segments
    def _use_segment(self, index: int) -> None:
        """Point every path at the segment currently being processed."""
        self.segment_index = index
        multi = len(getattr(self, "segments", [(0, 0)])) > 1
        base = self.dir / ("seg%03d" % index) if multi else self.dir
        self.segment_dir = base
        self.data_dir = base / "data" / "SoccerNetGS"
        self.split_dir = self.data_dir / "test"
        self.segment_clip = "SNGS-%05d" % (int(self.clip_name.split("-")[-1]) + index)             if multi else self.clip_name
        self.clip_dir = self.split_dir / self.segment_clip
        self.img_dir = self.clip_dir / "img1"
        self.frames_out = base / "annotated"
        self.config_path = base / "config.yaml"
        self.segment_video = base / "segment.mp4"

    @property
    def segment_bounds(self) -> tuple:
        return self.segments[self.segment_index]

    @property
    def segment_frames(self) -> int:
        start, end = self.segment_bounds
        return max(1, int(round((end - start) * self.fps)))

    # ---------------------------------------------------------------- helpers
    def say(self, line: str) -> None:
        with self._lock:
            self.log.append(line.rstrip())
            if len(self.log) > MAX_LOG_LINES:
                del self.log[: len(self.log) - MAX_LOG_LINES]

    def tail(self, n: int = 120) -> list[str]:
        with self._lock:
            return self.log[-n:]

    @property
    def elapsed(self) -> float:
        if not self.started_at:
            return 0.0
        return (self.finished_at or time.time()) - self.started_at

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "status": self.status,
            "video_name": self.video.get("name"),
            "video_id": self.video.get("id"),
            "start": self.start,
            "end": self.end,
            "fps": self.fps,
            "jersey_mode": self.jersey_mode,
            "resolution": self.resolution,
            "detector": self.detector,
            "detector_size": self.detector_size,
            "jersey_stride": self.jersey_stride,
            "calib_stride": self.calib_stride,
            "jersey_reader": self.jersey_reader,
            "segment_seconds": self.segment_seconds,
            "roster": self.roster,
            "render_video": self.render_video,
            "render_style": self.render_style,
            "analytics": self.analytics,
            "segments": len(self.segments),
            "segment_index": self.segment_index,
            "clip_name": self.clip_name,
            "stage": self.stage_key,
            "stage_title": self.stage_title,
            "stage_progress": round(self.stage_progress, 4),
            "progress": round(self.progress, 4),
            "expected_frames": self.expected_frames,
            "error": self.error,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
            "elapsed": round(self.elapsed, 1),
            "timings": {k: round(v, 1) for k, v in self.timings.items()},
            "has_result": self.result_path.exists(),
            "has_stats": (self.dir / "player_stats.json").exists(),
            "stages": [{"key": s.key, "title": s.title, "weight": s.weight}
                       for s in stages_for(self)],
        }

    @classmethod
    def from_disk(cls, data: dict) -> "Job":
        """Rebuild a finished job so history survives a server restart."""
        job = cls(
            video={"id": data.get("video_id"), "name": data.get("video_name"), "path": ""},
            start=data.get("start", 0.0), end=data.get("end", 0.0),
            fps=data.get("fps", 25), jersey_mode=data.get("jersey_mode", "CLIP"),
            resolution=data.get("resolution", "1080"),
            detector=data.get("detector", "rfdetr"),
            detector_size=data.get("detector_size", "medium"),
            jersey_stride=data.get("jersey_stride", 2),
            calib_stride=data.get("calib_stride", 1),
            jersey_reader=data.get("jersey_reader", "clip"),
            segment_seconds=data.get("segment_seconds", SEGMENT_SECONDS),
            roster=data.get("roster", ""),
            render_video=data.get("render_video", True),
            render_style=data.get("render_style", "fifa"),
            analytics=data.get("analytics", True),
            job_id=data["id"], clip_name=data.get("clip_name"),
        )
        job.status = "error" if data.get("status") == "running" else data.get("status", "done")
        job.error = data.get("error") or ("прогон прерван перезапуском сервера"
                                          if data.get("status") == "running" else "")
        job.progress = data.get("progress", 0.0)
        job.stage_key = data.get("stage", "")
        job.stage_title = data.get("stage_title", "")
        job.created_at = data.get("created_at", time.time())
        job.finished_at = data.get("finished_at")
        job.timings = data.get("timings") or {}
        job.log = list(data.get("log") or [])
        return job

    def persist(self) -> None:
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            data = self.as_dict()
            data["log"] = self.tail(200)
            payload = json.dumps(data, ensure_ascii=False, indent=2)
            (self.dir / "job.json").write_text(payload, encoding="utf-8")
        except Exception:
            pass

    def cancel(self) -> None:
        """Stop the stage's processes, and their children with them.

        A stage spawns a pool of workers, so killing only the process we
        launched leaves the pool running and the GPU held.  Windows needs
        `taskkill /T` for that; on Linux the same is done by signalling the
        process group, and the container has no `taskkill` at all — calling it
        there raised FileNotFoundError out of the endpoint even though the job
        did stop.
        """
        self._cancel = True
        for proc in list(self._procs):
            if proc.poll() is not None:
                continue
            try:
                if os.name == "nt":
                    subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                                   capture_output=True)
                else:
                    import signal
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                    except (ProcessLookupError, PermissionError):
                        proc.terminate()
            except Exception:
                # cancellation is best-effort: the job is already flagged, and
                # a failure to kill must not surface as a 500
                pass


# ------------------------------------------------------------------ stage defs
def _count_files(path: Path, suffix: str) -> int:
    try:
        return sum(1 for f in os.listdir(path) if f.lower().endswith(suffix))
    except OSError:
        return 0


def _extract_cmd(job: Job) -> list:
    scale = {"1080": "1920:1080", "720": "1280:720", "native": None}[job.resolution]
    vf = "fps=%d" % job.fps
    if scale:
        vf += ",scale=%s:flags=bicubic" % scale
    seg_start, seg_end = job.segment_bounds
    return [
        "ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "warning", "-y",
        "-ss", "%.3f" % seg_start,
        "-t", "%.3f" % max(0.04, seg_end - seg_start),
        "-i", str(job.video["path"]),
        "-vf", vf, "-q:v", "2", "-start_number", "1",
        str(job.img_dir / "%06d.jpg"),
    ]


def _encode_cmd(job: Job) -> list:
    return [
        "ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "warning", "-y",
        "-framerate", str(job.fps), "-start_number", "1",
        "-i", str(job.frames_out / "%06d.jpg"),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(job.segment_video),
    ]


def _absolutise_weights(node) -> None:
    """Point every `checkpoints/...` path at the mounted weights directory.

    The engine's config ships relative paths that resolve against its own
    folder.  The core keeps weights outside the image — they are gigabytes and
    never change — so each one is rewritten in place before the config is handed
    to a stage.  Missing files are left as they are: the stage's own error
    message names the model, which is more use than a path rewritten to nowhere.
    """
    if isinstance(node, dict):
        items = node.items()
    elif isinstance(node, list):
        items = enumerate(node)
    else:
        return
    for key, value in items:
        if isinstance(value, (dict, list)):
            _absolutise_weights(value)
        elif isinstance(value, str) and value.startswith("checkpoints/"):
            node[key] = str(CHECKPOINT_DIR / value[len("checkpoints/"):])
        elif isinstance(value, str) and value.lower().endswith(".gguf"):
            # The VLM paths in the engine config are absolute and point at
            # whatever machine the weights were first found on.  Only the file
            # name survives a move, so it is matched inside the weights
            # directory; if it is not there, the path is left alone and the
            # reader reports the model as unavailable rather than silently
            # reading nothing.
            candidate = CHECKPOINT_DIR / os.path.basename(value)
            if candidate.is_file():
                node[key] = str(candidate)


def _py(script: str, *args: str) -> list:
    return [str(VENV_PYTHON), script, *args]


STAGES: list[Stage] = [
    Stage("extract", "Извлечение кадров", 0.04, [
        Part("", _extract_cmd,
             total=lambda j: j.segment_frames,
             count=lambda j: _count_files(j.img_dir, ".jpg")),
    ]),
    Stage("analyze", "Калибровка поля ‖ детекция, трекинг, номера", 0.62, [
        Part("калибровка",
             lambda j: _py(str(RUNNERS / "run_kpts.py"),
                           "--split-dir", str(j.split_dir),
                           "--result-dir", str(j.dir / "kpts_results"),
                           "--workers", str(calib_workers(j)),
                           "--stride", str(j.calib_stride)),
             total=lambda j: max(1, _count_files(j.img_dir, ".jpg") // max(1, j.calib_stride)),
             count=lambda j: _count_files(j.img_dir, ".npy")),
        Part("трекинг",
             lambda j: _py("inference_soccernetGSR.py", "--config", str(j.config_path))),
    ]),
    Stage("idatr", "Сшивка треков и проекция на поле", 0.18, [
        Part("", lambda j: _py(str(RUNNERS / "run_idatr.py"), "--config", str(j.config_path))),
    ]),
    Stage("visualize", "Отрисовка разметки", 0.10, [
        Part("", lambda j: _py(str(RUNNERS / "run_visualize.py"),
                               "--clip-dir", str(j.clip_dir),
                               "--out-dir", str(j.frames_out),
                               "--style", j.render_style,
                               "--workers", str(VIS_WORKERS)),
             total=lambda j: _count_files(j.img_dir, ".jpg"),
             count=lambda j: _count_files(j.frames_out, ".jpg")),
    ]),
    Stage("encode", "Сборка видео", 0.06, [Part("", _encode_cmd)]),
]

RENDER_STAGES = ("visualize", "encode")


def stages_for(job: "Job") -> list:
    """The stages this job will actually run, with the weights re-normalised.

    Skipping rendering must not leave the progress bar stalling at 84%, so the
    remaining stages absorb the freed weight instead of the total shrinking.
    """
    if getattr(job, "render_video", True):
        return STAGES
    kept = [s for s in STAGES if s.key not in RENDER_STAGES]
    total = sum(s.weight for s in kept) or 1.0
    return [Stage(s.key, s.title, s.weight / total, s.parts) for s in kept]


class JobManager:
    def __init__(self) -> None:
        self.jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._restore()

    def _restore(self) -> None:
        for meta in sorted(JOB_DIR.glob("*/job.json")):
            try:
                data = json.loads(meta.read_text(encoding="utf-8"))
                self.jobs[data["id"]] = Job.from_disk(data)
            except Exception:
                continue

    def create(self, **kwargs) -> Job:
        job = Job(**kwargs)
        with self._lock:
            self.jobs[job.id] = job
        threading.Thread(target=self._run, args=(job,), daemon=True).start()
        return job

    def get(self, job_id: str) -> Optional[Job]:
        return self.jobs.get(job_id)

    def list(self) -> list[dict]:
        return sorted((j.as_dict() for j in self.jobs.values()),
                      key=lambda d: d["created_at"], reverse=True)

    # ------------------------------------------------------------------ runner
    def _prepare(self, job: Job) -> None:
        job.img_dir.mkdir(parents=True, exist_ok=True)
        job.frames_out.mkdir(parents=True, exist_ok=True)

        cfg = yaml.safe_load((REPO / "configs" / "config.yaml").read_text(encoding="utf-8"))
        _absolutise_weights(cfg)
        cfg["DATA_DIR"] = str(job.data_dir)
        cfg["IMG_SAVE_DIR"] = str(job.dir / "predicted")
        cfg["LOG_DIR"] = str(job.dir / "court")
        cfg["JERSEY_MODE"] = job.jersey_mode
        cfg["DATA_SETS"] = ["test"]
        cfg["TRACKER"]["FPS"] = job.fps
        cfg["JERSEY_STRIDE"] = job.jersey_stride
        cfg["JERSEY_READER"] = job.jersey_reader
        if job.roster:
            cfg["ROSTER"] = job.roster
        cfg.setdefault("DETECTOR", {})
        cfg["DETECTOR"]["TYPE"] = job.detector
        cfg["DETECTOR"]["RFDETR_MODEL"] = job.detector_size
        job.config_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True),
                                   encoding="utf-8")

    def _run_stage(self, job: Job, stage: Stage, base: float) -> None:
        job.stage_key = stage.key
        job.stage_title = stage.title
        job.stage_progress = 0.0
        job.say("== %s ==" % stage.title)
        started = time.perf_counter()

        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        # the engine, the vendored re-id package and the core itself all have to
        # be importable from a stage that runs with cwd set to the engine
        env["PYTHONPATH"] = os.pathsep.join(
            str(p) for p in (ROOT, REPO, REPO / "reid"))
        # runners that load their own weights read this rather than guessing a
        # path relative to whatever directory they happen to start in
        env["FG_CHECKPOINT_DIR"] = str(CHECKPOINT_DIR)

        shares = [0.0] * len(stage.parts)
        stop = threading.Event()
        failures: list[str] = []
        procs: list[subprocess.Popen] = []

        def publish() -> None:
            job.stage_progress = min(1.0, sum(shares) / len(shares))
            span = 1.0 / len(job.segments)
            done = job.segment_index * span
            job.progress = min(0.999, done + span * (base + stage.weight * job.stage_progress))

        def pump(index: int, part: Part, proc: subprocess.Popen) -> None:
            prefix = ("[%s] " % part.label) if part.label else ""
            assert proc.stdout is not None
            for raw in proc.stdout:
                line = raw.rstrip("\r\n").split("\r")[-1]
                if not line.strip():
                    continue
                job.say(prefix + line)
                if part.count is None and not DOWNLOAD_RE.search(line):
                    m = TQDM_RE.search(line)
                    if m and int(m.group(2)) > 0:
                        shares[index] = min(1.0, int(m.group(1)) / int(m.group(2)))
                        publish()
            code = proc.wait()
            if code != 0 and not job._cancel:
                failures.append("«%s%s» → код %d" % (prefix, stage.title, code))
            # the parts of a stage run side by side, so the stage costs what its
            # slowest part costs -- worth recording which one that was
            if part.label:
                job.timings["%s:%s" % (stage.key, part.label)] = round(
                    time.perf_counter() - started, 1)
            shares[index] = 1.0
            publish()

        def watch() -> None:
            while not stop.wait(0.5):
                for i, part in enumerate(stage.parts):
                    if part.count and part.total:
                        try:
                            shares[i] = min(1.0, part.count(job) / max(1, part.total(job)))
                        except Exception:
                            pass
                publish()

        pumps = []
        for index, part in enumerate(stage.parts):
            proc = subprocess.Popen(
                part.build(job), cwd=str(REPO), env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", bufsize=1,
                # own process group on POSIX, so cancelling can signal the
                # stage's whole worker pool rather than just the process we
                # launched — otherwise the pool keeps the GPU after a cancel
                start_new_session=(os.name != "nt"),
            )
            procs.append(proc)
            job._procs.append(proc)
            thread = threading.Thread(target=pump, args=(index, part, proc), daemon=True)
            thread.start()
            pumps.append(thread)

        watcher = threading.Thread(target=watch, daemon=True)
        watcher.start()
        for thread in pumps:
            thread.join()
        stop.set()
        watcher.join(timeout=2)
        for proc in procs:
            if proc in job._procs:
                job._procs.remove(proc)

        if job._cancel:
            raise RuntimeError("отменено пользователем")
        if failures:
            raise RuntimeError("; ".join(failures))

        job.timings[stage.key] = job.timings.get(stage.key, 0.0) + (time.perf_counter() - started)
        job.say("== %s: %.1f с ==" % (stage.title, time.perf_counter() - started))
        job.stage_progress = 1.0
        span = 1.0 / len(job.segments)
        job.progress = min(0.999, job.segment_index * span + span * (base + stage.weight))

    def _analyse(self, job: Job) -> None:
        """Build the event ledger and the per-player report from the run.

        Runs in-process: it loads no model and takes well under a second even
        for a full match, so a failure here must never cost the perception work
        that just finished.
        """
        job.stage_key = "analytics"
        job.stage_title = "Событийный журнал и статистика"
        try:
            from .analytics.run import analyse
            report = analyse(str(job.dir), kickoff_offset_s=job.kickoff_offset_s,
                             period=job.period, verbose=False)
            events = len(json.loads((job.dir / "events.json").read_text(
                encoding="utf-8"))["events"])
            named = sum(1 for p in report["players"].values() if p.get("name"))
            job.say("Статистика: %d событий, %d игроков (%d с именем) → "
                    "events.json, player_stats.json/.csv, events.sqlite"
                    % (events, len(report["players"]), named))
        except Exception as exc:
            job.say("Статистика не собрана: %s" % exc)

    def _run(self, job: Job) -> None:
        job.status = "running"
        job.started_at = time.time()
        job.persist()
        try:
            parts, jsons, skipped = [], [], []
            frame_offset = track_offset = 0
            total = len(job.segments)
            if total > 1:
                job.say("Интервал разбит на %d сегментов по ~%.0f с: кадры каждого "
                        "удаляются после обработки" % (total, job.segment_seconds))

            for index in range(total):
                if job._cancel:
                    raise RuntimeError("отменено пользователем")
                job._use_segment(index)
                seg_start, seg_end = job.segment_bounds
                if total > 1:
                    job.say("── сегмент %d/%d: %.1f–%.1f с ──"
                            % (index + 1, total, seg_start, seg_end))
                self._prepare(job)

                failure = None
                for attempt in (1, 2):
                    try:
                        base = 0.0
                        for stage in stages_for(job):
                            if job._cancel:
                                raise RuntimeError("отменено пользователем")
                            self._run_stage(job, stage, base)
                            base += stage.weight
                            job.persist()
                        if job.render_video and not job.segment_video.exists():
                            raise RuntimeError("сегмент не был собран")
                        failure = None
                        break
                    except Exception as exc:
                        if job._cancel:
                            raise
                        failure = exc
                        if attempt == 1:
                            # most segment failures are memory pressure from the
                            # parallel calibration workers: retry single-threaded
                            job.calib_workers_override = 1
                            job.say("Сегмент %d: повтор в один поток калибровки (%s)"
                                    % (index + 1, exc))
                        job.persist()
                job.calib_workers_override = None

                if failure is not None:
                    # one bad segment must not throw away hours of work on the
                    # rest of the match: note it and carry on
                    skipped.append(index + 1)
                    job.say("СЕГМЕНТ %d ПРОПУЩЕН: %s" % (index + 1, failure))
                    frame_offset += job.segment_frames
                    self._drop_frames(job)
                    continue

                if job.render_video:
                    parts.append(job.segment_video)
                jsons.append((job.clip_dir / ("%s.json" % job.segment_clip),
                              frame_offset, track_offset))
                frame_offset += _count_files(job.img_dir, ".jpg")
                track_offset += self._max_track_id(job) + 1
                self._drop_frames(job)

            job.stage_key = "assemble"
            job.stage_title = "Сборка результата"
            if not jsons:
                raise RuntimeError("ни один сегмент не удалось обработать")
            if skipped:
                job.say("Пропущено сегментов: %d (%s)"
                        % (len(skipped), ", ".join(map(str, skipped))))
            if job.render_video:
                concat_videos(parts, job.result_path)
            stats = merge_predictions(jsons, job.predictions_path)
            job.say("Итог: %d предсказаний, %d треков"
                    % (stats["predictions"], stats["tracks"]))

            if job.render_video and not job.result_path.exists():
                raise RuntimeError("видео с разметкой не было создано")

            if job.analytics:
                self._analyse(job)

            job.status = "done"
            job.progress = 1.0
            job.stage_key = "done"
            job.stage_title = "Готово"
            job.say("Готово за %.1f с" % job.elapsed)
        except Exception as exc:
            job.status = "canceled" if job._cancel else "error"
            job.error = str(exc)
            job.say("ОШИБКА: %s" % exc)
        finally:
            job.finished_at = time.time()
            job.persist()
            self._cleanup(job)

    @staticmethod
    def _max_track_id(job: Job) -> int:
        """Largest track id of the finished segment, to keep ids unique."""
        path = job.clip_dir / ("refined_%s.txt" % job.segment_clip)
        best = 0
        try:
            with path.open(encoding="utf-8") as fh:
                for line in fh:
                    parts = line.split(",")
                    if len(parts) > 1:
                        best = max(best, int(float(parts[1])))
        except Exception:
            pass
        return best

    @staticmethod
    def _drop_frames(job: Job) -> None:
        """Frames are the bulk of the footprint: remove them once rendered."""
        shutil.rmtree(job.img_dir, ignore_errors=True)
        shutil.rmtree(job.frames_out, ignore_errors=True)

    @staticmethod
    def _cleanup(job: Job) -> None:
        """Frames and per-segment videos are redundant once the result exists."""
        if job.status != "done":
            return
        for index in range(len(job.segments)):
            job._use_segment(index)
            JobManager._drop_frames(job)
            if len(job.segments) > 1:
                job.segment_video.unlink(missing_ok=True)


manager = JobManager()
