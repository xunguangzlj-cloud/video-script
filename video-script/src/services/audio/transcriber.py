"""语音转录器 — faster-whisper 语音识别 + 时间戳"""

from pathlib import Path
from typing import Optional

from src.config.settings import Settings


class WhisperTranscriber:
    """faster-whisper 语音转文字"""

    def __init__(self, model_size: str = "medium", device: str = "auto"):
        """
        Args:
            model_size: 模型大小 (tiny/base/small/medium/large-v3)
            device: 运行设备 (auto/cpu/cuda)
        """
        self.model_size = model_size
        self.device = device
        self._model = None

    def _load_model(self):
        """延迟加载模型"""
        if self._model is not None:
            return

        from faster_whisper import WhisperModel

        # 自动检测设备
        if self.device == "auto":
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
                compute_type = "float16" if device == "cuda" else "int8"
            except Exception:
                device = "cpu"
                compute_type = "int8"
        else:
            device = self.device
            compute_type = "float16" if device == "cuda" else "int8"

        print(f"   加载 faster-whisper 模型: {self.model_size} ({device}/{compute_type})")
        self._model = WhisperModel(
            self.model_size,
            device=device,
            compute_type=compute_type,
            download_root=str(Path.home() / ".cache" / "faster-whisper"),
        )
        print(f"   模型加载完成")

    def transcribe(
        self,
        audio_path: str | Path,
        language: Optional[str] = "zh",
    ) -> str:
        """转录音频文件为文本

        Args:
            audio_path: WAV 音频文件路径（16kHz, 单声道, 16-bit PCM）
            language: 语言代码（zh=中文, 留空则自动检测）

        Returns:
            带时间戳的转录文本，每行格式： [HH:MM:SS] 文本
        """
        audio_path = Path(audio_path)
        self._load_model()

        segments, info = self._model.transcribe(
            str(audio_path),
            language=language,
            beam_size=5,
            vad_filter=True,         # 自动过滤非语音段
            vad_parameters=dict(     # VAD 参数
                min_silence_duration_ms=500,
                threshold=0.5,
            ),
        )

        lines = []
        for segment in segments:
            ts = self._format_timestamp(segment.start)
            text = segment.text.strip()
            if text:
                lines.append(f"[{ts}] {text}")

        result = "\n".join(lines)

        # 打印统计
        total_duration = info.duration if hasattr(info, 'duration') else 0
        print(f"   语言: {info.language} (概率: {info.language_probability:.2%})")
        print(f"   音频时长: {total_duration:.1f}s")
        print(f"   转录段数: {len(lines)}")
        print(f"   总字数: {len(result)}")

        return result

    def transcribe_plain(self, audio_path: str | Path, language: Optional[str] = "zh") -> str:
        """转录音频，返回纯文本（不带时间戳）"""
        audio_path = Path(audio_path)
        self._load_model()

        segments, _ = self._model.transcribe(
            str(audio_path),
            language=language,
            beam_size=5,
            vad_filter=True,
        )

        lines = [seg.text.strip() for seg in segments if seg.text.strip()]
        return "\n".join(lines)

    @staticmethod
    def _format_timestamp(seconds: float) -> str:
        """将秒数格式化为 HH:MM:SS"""
        h, m = divmod(int(seconds), 3600)
        m, s = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"
