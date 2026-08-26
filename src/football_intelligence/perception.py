"""HTTP adapter for the existing, measured Football Core perception service."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import httpx


class LegacyCoreClient:
    def __init__(self, base_url: str, api_key: str = "", timeout_s: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.headers = {"X-API-Key": api_key} if api_key else {}
        self.timeout_s = timeout_s

    def health(self) -> dict[str, Any]:
        response = httpx.get(f"{self.base_url}/v1/health", headers=self.headers, timeout=10)
        response.raise_for_status()
        return response.json()

    def upload_video(self, video_path: str | Path) -> dict[str, Any]:
        path = Path(video_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open("rb") as handle:
            response = httpx.post(
                f"{self.base_url}/v1/videos",
                headers=self.headers,
                files={"file": (path.name, handle, "application/octet-stream")},
                timeout=None,
            )
        response.raise_for_status()
        return response.json()

    def create_job(self, payload: dict[str, Any]) -> dict[str, Any]:
        response = httpx.post(
            f"{self.base_url}/v1/jobs",
            headers=self.headers,
            json=payload,
            timeout=self.timeout_s,
        )
        response.raise_for_status()
        return response.json()

    def wait_job(
        self,
        job_id: str,
        poll_seconds: float = 5.0,
        deadline_seconds: float = 24 * 3600,
        progress_callback=None,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + deadline_seconds
        while time.monotonic() < deadline:
            response = httpx.get(
                f"{self.base_url}/v1/jobs/{job_id}", headers=self.headers, timeout=self.timeout_s
            )
            response.raise_for_status()
            job = response.json()
            if progress_callback:
                progress_callback(job)
            if job["status"] == "done":
                return job
            if job["status"] in {"error", "canceled"}:
                raise RuntimeError(
                    f"perception job {job_id} ended as {job['status']}: {job.get('error', '')}"
                )
            time.sleep(poll_seconds)
        raise TimeoutError(f"perception job {job_id} exceeded {deadline_seconds}s")

    def artifact(self, job_id: str, name: str) -> dict[str, Any]:
        endpoints = {"events": "events", "predictions": "predictions", "players": "players"}
        if name not in endpoints:
            raise ValueError(f"unknown legacy artifact {name}")
        response = httpx.get(
            f"{self.base_url}/v1/jobs/{job_id}/{endpoints[name]}",
            headers=self.headers,
            timeout=self.timeout_s,
        )
        response.raise_for_status()
        return response.json()

    def download_artifact(self, job_id: str, name: str, target: str | Path) -> Path:
        endpoints = {"video": "video", "players_csv": "players.csv"}
        if name not in endpoints:
            raise ValueError(f"unknown downloadable legacy artifact {name}")
        destination = Path(target)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        with httpx.stream(
            "GET",
            f"{self.base_url}/v1/jobs/{job_id}/{endpoints[name]}",
            headers=self.headers,
            timeout=None,
        ) as response:
            response.raise_for_status()
            with temporary.open("wb") as output:
                for chunk in response.iter_bytes(1024 * 1024):
                    output.write(chunk)
        temporary.replace(destination)
        return destination

    def analyse_video(
        self,
        video_path: str | Path,
        start_s: float,
        end_s: float,
        fps: int = 12,
        roster: dict | None = None,
        render_video: bool = True,
        progress_callback=None,
    ) -> dict[str, Any]:
        video = self.upload_video(video_path)
        request: dict[str, Any] = {
            "video_id": video["id"],
            "start": start_s,
            "end": end_s,
            "fps": fps,
            "render_video": render_video,
            "analytics": True,
        }
        if roster:
            request["roster"] = roster
        job = self.create_job(request)
        settled = self.wait_job(job["id"], progress_callback=progress_callback)
        return {
            "job": settled,
            "events": self.artifact(job["id"], "events"),
            "predictions": self.artifact(job["id"], "predictions"),
            "players": self.artifact(job["id"], "players"),
        }
