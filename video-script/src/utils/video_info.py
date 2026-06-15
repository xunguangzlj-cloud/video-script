"""视频元数据获取 — 封装 ffprobe"""

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

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
    """使用 ffprobe 获取视频元数据"""
    filepath = Path(filepath)

    info = VideoInfo(filepath=str(filepath))

    try:
        cmd = [
            Settings.FFPROBE_PATH,
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            str(filepath),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        data = json.loads(result.stdout)

        fmt = data.get("format", {})

        info.duration_seconds = float(fmt.get("duration", 0))
        info.bitrate = int(fmt.get("bit_rate", 0))

        subtitles = []
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
                subtitles.append(stream)

        # 字幕检测
        if subtitles:
            info.has_subtitle_stream = True
            # 判断软字幕类型
            codecs = [s.get("codec_name", "") for s in subtitles]
            if "mov_text" in codecs or "subrip" in codecs or "ass" in codecs:
                info.subtitle_type = "soft"
            else:
                info.subtitle_type = "soft"  # 有字幕流就算是软的
        else:
            info.subtitle_type = "unknown"  # 可能是硬字幕或没有

    except Exception as e:
        print(f"获取视频信息失败: {e}")

    return info
