from __future__ import annotations

import argparse
import cgi
import json
import mimetypes
import os
import re
import shutil
import sys
import threading
import time
import uuid
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import unquote, urlparse

from pipeline import PipelineError, run_pipeline


ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "static"
JOBS_DIR = ROOT / "workspace" / "jobs"
MAX_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024
VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".webm", ".mkv"}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg"}


def safe_filename(value: str, fallback: str) -> str:
    name = Path(value or "").name
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    return stem[:120] or fallback


class JobStore:
    def __init__(self) -> None:
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def create(self) -> Dict[str, Any]:
        job_id = uuid.uuid4().hex[:12]
        job = {
            "id": job_id,
            "status": "queued",
            "progress": 0,
            "message": "任务已创建",
            "created_at": time.time(),
            "result": None,
            "error": None,
        }
        with self._lock:
            self._jobs[job_id] = job
        return dict(job)

    def update(self, job_id: str, **changes: Any) -> None:
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].update(changes)

    def get(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            job = self._jobs.get(job_id)
            return dict(job) if job else None

    def delete(self, job_id: str) -> None:
        with self._lock:
            self._jobs.pop(job_id, None)


JOBS = JobStore()


def _run_job(job_id: str, settings: Dict[str, Any]) -> None:
    job_dir = JOBS_DIR / job_id

    def progress(value: int, message: str) -> None:
        JOBS.update(job_id, status="running", progress=value, message=message)

    try:
        result = run_pipeline(
            video_paths=settings["video_paths"],
            job_dir=job_dir,
            brief=settings["brief"],
            target_duration=settings["target_duration"],
            style=settings["style"],
            language=settings["language"],
            use_ai=settings["use_ai"],
            music_path=settings.get("music_path"),
            width=settings["width"],
            height=settings["height"],
            progress=progress,
        )
        result["urls"] = {
            key: f"/files/{job_id}/{filename}"
            for key, filename in {
                "video": result["video"],
                "plan": result["plan"],
                "manifest": result["manifest"],
                "captions": result["captions"],
            }.items()
        }
        JOBS.update(
            job_id,
            status="completed",
            progress=100,
            message="成片已完成",
            result=result,
        )
    except Exception as exc:
        JOBS.update(
            job_id,
            status="failed",
            message="处理失败",
            error=str(exc),
        )


class Handler(BaseHTTPRequestHandler):
    server_version = "VideoRebuilder/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stdout.write("[web] " + fmt % args + "\n")

    def _send_json(self, data: Dict[str, Any], status: int = 200) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _serve_path(self, path: Path, download: bool = False) -> None:
        if not path.exists() or not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        size = path.stat().st_size
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        range_header = self.headers.get("Range")
        start = 0
        end = size - 1
        status = HTTPStatus.OK
        if range_header:
            match = re.match(r"bytes=(\d*)-(\d*)", range_header)
            if match:
                if match.group(1):
                    start = int(match.group(1))
                if match.group(2):
                    end = min(int(match.group(2)), size - 1)
                if start > end or start >= size:
                    self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                    return
                status = HTTPStatus.PARTIAL_CONTENT
        length = end - start + 1
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("Accept-Ranges", "bytes")
        if status == HTTPStatus.PARTIAL_CONTENT:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        if download:
            self.send_header("Content-Disposition", f'attachment; filename="{path.name}"')
        self.end_headers()
        with path.open("rb") as handle:
            handle.seek(start)
            remaining = length
            while remaining:
                chunk = handle.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        if path == "/":
            self._serve_path(STATIC_DIR / "index.html")
            return
        if path == "/api/status":
            self._send_json(
                {
                    "ready": bool(shutil.which("ffmpeg") and shutil.which("ffprobe")),
                    "ffmpeg": bool(shutil.which("ffmpeg")),
                    "ai_configured": bool(os.environ.get("OPENAI_API_KEY")),
                    "model": os.environ.get("VIDEO_REBUILDER_MODEL", "gpt-5.6-sol"),
                }
            )
            return
        if path.startswith("/api/jobs/"):
            job_id = path.rsplit("/", 1)[-1]
            job = JOBS.get(job_id)
            if not job:
                self._send_json({"error": "任务不存在"}, 404)
            else:
                self._send_json(job)
            return
        if path.startswith("/files/"):
            parts = path.split("/")
            if len(parts) != 4:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            job_id, filename = parts[2], safe_filename(parts[3], "file")
            if not re.fullmatch(r"[0-9a-f]{12}", job_id):
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            job_root = (JOBS_DIR / job_id).resolve()
            file_path = (job_root / filename).resolve()
            if job_root not in file_path.parents:
                self.send_error(HTTPStatus.FORBIDDEN)
                return
            self._serve_path(file_path, download=parsed.query == "download=1")
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/jobs":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        content_length = int(self.headers.get("Content-Length") or 0)
        if content_length <= 0 or content_length > MAX_UPLOAD_BYTES:
            self._send_json({"error": "上传内容为空或超过 2 GB"}, 413)
            return
        content_type = self.headers.get("Content-Type") or ""
        if "multipart/form-data" not in content_type:
            self._send_json({"error": "请使用 multipart/form-data 上传"}, 400)
            return
        job: Optional[Dict[str, Any]] = None
        job_dir: Optional[Path] = None
        try:
            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={
                    "REQUEST_METHOD": "POST",
                    "CONTENT_TYPE": content_type,
                    "CONTENT_LENGTH": str(content_length),
                },
            )
            video_fields = form["videos"] if "videos" in form else []
            if not isinstance(video_fields, list):
                video_fields = [video_fields]
            video_fields = [field for field in video_fields if getattr(field, "filename", None)]
            if not video_fields:
                self._send_json({"error": "请至少选择一个视频"}, 400)
                return

            prepared_videos = []
            for index, field in enumerate(video_fields[:8]):
                filename = safe_filename(field.filename, f"video_{index}.mp4")
                if Path(filename).suffix.lower() not in VIDEO_EXTENSIONS:
                    raise PipelineError(f"不支持的视频格式：{filename}")
                prepared_videos.append((field, filename))

            music_field = None
            music_filename = None
            if "music" in form and getattr(form["music"], "filename", None):
                music_field = form["music"]
                music_filename = safe_filename(music_field.filename, "music.mp3")
                if Path(music_filename).suffix.lower() not in AUDIO_EXTENSIONS:
                    raise PipelineError(f"不支持的音乐格式：{music_filename}")

            quality = str(form.getfirst("quality", "final"))
            width, height = (720, 1280) if quality == "preview" else (1080, 1920)
            target_duration = max(
                3.0, min(60.0, float(form.getfirst("duration", "15")))
            )

            job = JOBS.create()
            job_dir = JOBS_DIR / job["id"]
            input_dir = job_dir / "inputs"
            input_dir.mkdir(parents=True, exist_ok=True)
            video_paths: List[Path] = []
            for index, (field, filename) in enumerate(prepared_videos):
                destination = input_dir / f"{index:02d}_{filename}"
                with destination.open("wb") as handle:
                    shutil.copyfileobj(field.file, handle, length=1024 * 1024)
                video_paths.append(destination)

            music_path: Optional[Path] = None
            if music_field is not None and music_filename:
                music_path = input_dir / music_filename
                with music_path.open("wb") as handle:
                    shutil.copyfileobj(music_field.file, handle, length=1024 * 1024)

            settings = {
                "video_paths": video_paths,
                "music_path": music_path,
                "brief": str(form.getfirst("brief", "")),
                "target_duration": target_duration,
                "style": str(form.getfirst("style", "ugc")),
                "language": str(form.getfirst("language", "简体中文")),
                "use_ai": str(form.getfirst("use_ai", "true")).lower() == "true",
                "width": width,
                "height": height,
            }
            thread = threading.Thread(target=_run_job, args=(job["id"], settings), daemon=True)
            thread.start()
            self._send_json({"job_id": job["id"], "status": "queued"}, 202)
        except Exception as exc:
            if job and job_dir:
                JOBS.delete(job["id"])
                shutil.rmtree(job_dir, ignore_errors=True)
            self._send_json({"error": str(exc)}, 400)


def serve(host: str, port: int, open_browser: bool) -> None:
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((host, port), Handler)
    url = f"http://{host}:{port}"
    print(f"AI 视频重构器已启动：{url}")
    print("按 Control-C 停止。")
    if open_browser:
        threading.Timer(0.7, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n服务已停止。")
    finally:
        server.server_close()


def render_cli(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).resolve()

    def progress(value: int, message: str) -> None:
        print(f"[{value:3d}%] {message}")

    result = run_pipeline(
        [Path(value).resolve() for value in args.videos],
        output_dir,
        brief=args.brief,
        target_duration=args.duration,
        style=args.style,
        language=args.language,
        use_ai=not args.no_ai,
        music_path=Path(args.music).resolve() if args.music else None,
        width=720 if args.preview else 1080,
        height=1280 if args.preview else 1920,
        progress=progress,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="AI 视频二次重构器")
    subparsers = parser.add_subparsers(dest="command")
    server_parser = subparsers.add_parser("serve", help="启动本地操作页")
    server_parser.add_argument("--host", default="127.0.0.1")
    server_parser.add_argument("--port", type=int, default=8765)
    server_parser.add_argument("--no-browser", action="store_true")

    render_parser = subparsers.add_parser("render", help="从命令行直接生成成片")
    render_parser.add_argument("videos", nargs="+")
    render_parser.add_argument("--brief", default="前三秒展示结果，再给出细节证明，最后明确行动提示。")
    render_parser.add_argument("--duration", type=float, default=15)
    render_parser.add_argument("--style", choices=["ugc", "product-demo", "fast-cut", "story"], default="ugc")
    render_parser.add_argument("--language", default="简体中文")
    render_parser.add_argument("--music")
    render_parser.add_argument("--output-dir", default="workspace/cli-output")
    render_parser.add_argument("--preview", action="store_true")
    render_parser.add_argument("--no-ai", action="store_true")

    args = parser.parse_args()
    if args.command == "render":
        render_cli(args)
    else:
        if args.command is None:
            args.host, args.port, args.no_browser = "127.0.0.1", 8765, False
        serve(args.host, args.port, not args.no_browser)


if __name__ == "__main__":
    main()
