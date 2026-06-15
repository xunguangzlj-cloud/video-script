"""流水线编排器 — QThread 后台处理"""

import json
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QThread, Signal

from src.config.settings import Settings
from src.utils.video_info import get_video_info, VideoInfo


class PipelineWorker(QThread):
    """后台处理线程"""

    # 信号
    progress = Signal(int, str)       # (百分比, 状态描述)
    finished = Signal(dict)           # 完成的剧本 JSON
    error = Signal(str)               # 错误信息
    subtitle_found = Signal(str, str) # (字幕来源, 预览文本)

    def __init__(self, video_path: str, parent=None):
        super().__init__(parent)
        self.video_path = video_path
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        """执行完整流水线（将在后续阶段实现）"""
        try:
            # 阶段 1: 视频元数据
            self.progress.emit(5, "读取视频信息...")
            info = get_video_info(self.video_path)

            # 阶段 2: 字幕提取（三级回退）
            self.progress.emit(10, "提取字幕...")
            subtitle_text, subtitle_source = self._extract_subtitles(info)

            # 阶段 3: 关键帧提取
            self.progress.emit(20, "检测场景边界...")

            # 阶段 4: GLM-4V 视觉理解
            self.progress.emit(30, "AI 分析画面...")

            # 阶段 5: DeepSeek 剧本融合
            self.progress.emit(70, "AI 生成剧本...")

            # 完成
            self.progress.emit(100, "完成！")
            # self.finished.emit(script_json)

        except Exception as e:
            self.error.emit(str(e))

    def _extract_subtitles(self, info: VideoInfo) -> tuple[Optional[str], str]:
        """三级字幕提取"""
        from src.services.audio.subtitle_extractor import extract_subtitles

        text, source = extract_subtitles(self.video_path)

        if text:
            preview = text[:200] + "..." if len(text) > 200 else text
            self.subtitle_found.emit(source, preview)

        return text, source
