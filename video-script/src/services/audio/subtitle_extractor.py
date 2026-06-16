"""字幕提取器 — 三级策略：软字幕 → Whisper 语音识别（主） → 硬字幕 OCR（辅）"""

from pathlib import Path
from src.utils.ffmpeg_helper import extract_soft_subtitle, extract_audio
from src.services.audio.transcriber import WhisperTranscriber


def extract_subtitles(video_path: str | Path) -> tuple[str | None, str]:
    """三级字幕提取

    1. 优先软字幕（FFmpeg，2秒，100%准确）
    2. Whisper 语音识别（主方案，覆盖全程对话）
    3. 硬字幕 OCR（由调用方通过 GLM-4V 辅助）

    Args:
        video_path: 视频文件路径

    Returns:
        (字幕文本, 来源标记)
        来源: "soft" / "whisper" / "whisper+ocr" / "none"
    """
    video_path = Path(video_path)

    # 第一级：尝试提取软字幕
    subtitle_text = extract_soft_subtitle(video_path)
    if subtitle_text:
        return subtitle_text, "soft"

    # 第二级：Whisper 语音识别（主方案）
    return None, "whisper"


def extract_subtitles_via_whisper(video_path: str | Path) -> tuple[str | None, str]:
    """使用 Whisper 从音频中提取对话文本

    Returns:
        (转录文本, 来源标记)
    """
    video_path = Path(video_path)

    try:
        # 1. 提取音频
        print("   正在从视频提取音频...")
        audio_path = extract_audio(str(video_path))

        # 2. Whisper 转录
        print("   正在语音识别...")
        transcriber = WhisperTranscriber(model_size="medium")
        text = transcriber.transcribe(audio_path, language="zh")

        # 3. 清理临时音频文件
        if audio_path.exists():
            audio_path.unlink()

        if text.strip():
            return text, "whisper"
        else:
            return None, "none"

    except Exception as e:
        print(f"   Whisper 转录失败: {e}")
        return None, "whisper_failed"
