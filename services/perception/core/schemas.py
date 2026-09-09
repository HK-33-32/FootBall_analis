"""Request and response shapes for the HTTP API.

Kept in one place so the OpenAPI document the service generates is the contract,
and a client can be generated from it rather than written by hand.
"""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, model_validator

Fps = Literal[5, 10, 12, 25, 30]
Resolution = Literal["1080", "720", "native"]
Detector = Literal["rfdetr", "yolox"]
DetectorSize = Literal["nano", "small", "medium", "base", "large"]
JerseyReader = Literal["clip", "qwen-vl"]
RenderStyle = Literal["fifa", "boxes"]
JobStatus = Literal["queued", "running", "done", "error", "canceled"]


class AnalysisRequest(BaseModel):
    """What to analyse and how.

    Only `video_id` and the interval are required; every other field falls back
    to the deployment default, so a caller that does not care about detector
    sizes never has to name one.
    """
    video_id: str = Field(..., description="id returned by POST /v1/videos")
    start: float = Field(0.0, ge=0, description="seconds into the source video")
    end: float = Field(..., gt=0, description="seconds into the source video")

    fps: Optional[Fps] = Field(None, description="frames per second to analyse")
    resolution: Optional[Resolution] = None
    detector: Optional[Detector] = None
    detector_size: Optional[DetectorSize] = None
    jersey_reader: Optional[JerseyReader] = None
    jersey_stride: Optional[int] = Field(None, ge=1, le=10)
    calib_stride: Optional[int] = Field(None, ge=1, le=5)
    segment_seconds: Optional[float] = Field(None, ge=15, le=600)

    render_video: Optional[bool] = Field(
        None, description="draw an annotated video; off by default, statistics "
                          "are produced either way")
    render_style: Optional[RenderStyle] = None
    analytics: Optional[bool] = Field(
        None, description="build the event ledger and per-player report")

    roster: Optional[dict] = Field(
        None, description="squad lists; restricts numbers to those that exist "
                          "and supplies surnames")
    roster_id: Optional[str] = Field(
        None, description="id of a roster shipped with the deployment")
    kickoff_offset_s: Optional[float] = Field(
        None, description="where in the source video the period kicked off; "
                          "enables match-clock timestamps on every event")
    period: Optional[int] = Field(None, ge=1, le=5)

    callback_url: Optional[str] = Field(
        None, description="POSTed the finished job payload when it settles")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="opaque; returned untouched")

    @model_validator(mode="after")
    def _interval(self):
        if self.end - self.start < 0.2:
            raise ValueError("interval shorter than 0.2 s")
        return self


class StageInfo(BaseModel):
    key: str
    title: str
    weight: float


class JobSummary(BaseModel):
    id: str
    status: JobStatus
    progress: float = Field(..., ge=0, le=1)
    stage: str = ""
    stage_title: str = ""
    created_at: float
    elapsed: float = 0.0
    error: str = ""
    video_id: Optional[str] = None
    video_name: Optional[str] = None
    start: float = 0.0
    end: float = 0.0


class JobDetail(JobSummary):
    fps: int = 12
    resolution: str = "1080"
    detector: str = "rfdetr"
    detector_size: str = "large"
    jersey_reader: str = "qwen-vl"
    jersey_stride: int = 3
    calib_stride: int = 1
    segment_seconds: float = 60.0
    segments: int = 1
    segment_index: int = 0
    render_video: bool = False
    render_style: str = "fifa"
    analytics: bool = True
    expected_frames: int = 0
    has_result: bool = False
    has_stats: bool = False
    stages: list[StageInfo] = Field(default_factory=list)
    timings: dict[str, float] = Field(default_factory=dict)
    log: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class VideoInfo(BaseModel):
    id: str
    name: str
    size: int = 0
    duration: float = 0.0
    fps: float = 0.0
    width: int = 0
    height: int = 0


class RosterInfo(BaseModel):
    id: str
    name: str = ""
    teams: list[str] = Field(default_factory=list)


class HealthReport(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    gpu: Optional[str] = None
    cuda_available: bool = False
    ffmpeg: bool = False
    checkpoints: dict[str, bool] = Field(default_factory=dict)
    settings: dict[str, Any] = Field(default_factory=dict)
    queue: dict[str, int] = Field(default_factory=dict)
