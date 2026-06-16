"""视频元数据获取 — 封装 ffprobe + OpenCV 兜底"""

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2

from src.config.settings import Settings


@dataclass
class VideoInfo:
    """视频基本信息"""
    filepath: str
    duration_seconds: float = 0.0
    width: int = 0
    height: int = 0
    fps: float = 0.0
    bitrate: int = 0
    codec: str = ""
    has_subtitle_stream: bool = False
    subtitle_type: str = ""           # "soft" / "hard" / "none"
    has_audio_stream: bool = False

    @property
    def duration_minutes(self) -> float:
        return self.duration_seconds / 60.0

    @property
    def duration_str(self) -> str:
        mins, secs = divmod(int(self.duration_seconds), 60)
        hours, mins = divmod(mins, 60)
        return f"{hours:02d}:{mins:02d}:{secs:02d}"


def get_video_info(filepath: str | Path) -> VideoInfo:
    """使用 ffprobe 获取视频元数据，失败则回退到 OpenCV"""
    filepath = Path(filepath)

    info = VideoInfo(filepath=str(filepath))

    # 方法 1: ffprobe
    try:
        cmd = [
            Settings.FFPROBE_PATH,
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            str(filepath),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30,
                                encoding="utf-8", errors="replace")

        if result.stdout.strip():
            data = json.loads(result.stdout)

            fmt = data.get("format", {})

            info.duration_seconds = float(fmt.get("duration", 0))
            info.bitrate = int(fmt.get("bit_rate", 0))

            for stream in data.get("streams", []):
                codec_type = stream.get("codec_type", "")
                if codec_type == "video" and info.width == 0:
                    info.width = stream.get("width", 0)
                    info.height = stream.get("height", 0)
                    fps_parts = stream.get("r_frame_rate", "0/1").split("/")
                    if len(fps_parts) == 2 and fps_parts[1] != "0":
                        info.fps = float(fps_parts[0]) / float(fps_parts[1])
                    info.codec = stream.get("codec_name", "")
                elif codec_type == "audio":
                    info.has_audio_stream = True
                elif codec_type == "subtitle":
                    info.subtitle_type = "soft"
                    info.has_subtitle_stream = True

    except Exception as e:
        print(f"ffprobe 获取视频信息失败: {e}")

    # 方法 2: OpenCV 兜底获取基本参数
    if info.duration_seconds <= 0 or info.width == 0:
        try:
            cap = cv2.VideoCapture(str(filepath))
            if cap.isOpened():
                info.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                info.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                info.fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                if info.fps > 0 and frame_count > 0:
                    info.duration_seconds = frame_count / info.fps
                cap.release()
        except Exception as e:
            print(f"OpenCV 获取视频信息失败: {e}")

    return info
