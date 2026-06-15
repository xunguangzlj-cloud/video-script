"""字幕提取器 — 三级回退策略"""

from pathlib import Path
from src.config.settings import Settings
from src.utils.ffmpeg_helper import extract_soft_subtitle
from src.utils.video_info import VideoInfo


def extract_subtitles(video_path: str | Path) -> tuple[str | None, str]:
    """三级字幕提取策略

    1. 优先提取软字幕（FFmpeg，2秒，100%准确）
    2. 视频有硬字幕 → 标记由 GLM-4V OCR（不额外花钱）
    3. 都没有 → 标记回退到 Whisper

    Args:
        video_path: 视频文件路径

    Returns:
        (字幕文本, 来源标记)
        来源: "soft" / "hard_ocr" / "whisper" / "none"
    """
    video_path = Path(video_path)

    # 第一级：尝试提取软字幕
    subtitle_text = extract_soft_subtitle(video_path)
    if subtitle_text:
        return subtitle_text, "soft"

    # 第二级 & 第三级：由调用方决定（需要看实际视频帧）
    # 这里只返回标记
    return None, "unknown"
