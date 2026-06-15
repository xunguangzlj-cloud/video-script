"""FFmpeg 辅助工具"""

import subprocess
from pathlib import Path

from src.config.settings import Settings


def extract_audio(input_video: str | Path, output_audio: str | Path = None) -> Path:
    """从视频中提取音轨（用于 Whisper 兜底方案）

    Args:
        input_video: 视频文件路径
        output_audio: 输出音频路径（默认 video_name.wav）

    Returns:
        输出音频文件路径
    """
    input_video = Path(input_video)
    if output_audio is None:
        output_audio = input_video.with_suffix(".wav")
    else:
        output_audio = Path(output_audio)

    cmd = [
        Settings.FFMPEG_PATH,
        "-i", str(input_video),
        "-vn",                    # 不要视频
        "-acodec", "pcm_s16le",   # 16-bit PCM
        "-ar", "16000",           # 16kHz 采样率（Whisper 要求）
        "-ac", "1",               # 单声道
        "-y",                     # 覆盖已有文件
        str(output_audio),
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    return output_audio


def extract_soft_subtitle(input_video: str | Path) -> str | None:
    """从视频中提取软字幕

    Args:
        input_video: 视频文件路径

    Returns:
        字幕文本内容，或 None（如无软字幕）
    """
    input_video = Path(input_video)
    output_srt = input_video.with_suffix(".srt")

    try:
        cmd = [
            Settings.FFMPEG_PATH,
            "-i", str(input_video),
            "-map", "0:s:0",       # 第一个字幕流
            "-y",
            str(output_srt),
        ]
        subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=True)

        if output_srt.exists():
            text = output_srt.read_text(encoding="utf-8", errors="replace")
            output_srt.unlink()  # 清理临时文件
            return text if text.strip() else None
    except Exception:
        pass

    return None


def check_ffmpeg() -> bool:
    """检查 FFmpeg 是否可用"""
    try:
        subprocess.run(
            [Settings.FFMPEG_PATH, "-version"],
            capture_output=True,
            timeout=5,
            check=True,
        )
        return True
    except Exception:
        return False


def check_ffprobe() -> bool:
    """检查 ffprobe 是否可用"""
    try:
        subprocess.run(
            [Settings.FFPROBE_PATH, "-version"],
            capture_output=True,
            timeout=5,
            check=True,
        )
        return True
    except Exception:
        return False
