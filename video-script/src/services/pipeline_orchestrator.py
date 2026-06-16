"""流水线编排器 — QThread 后台处理"""

import json
import traceback
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QThread, Signal

from src.config.settings import Settings
from src.utils.video_info import get_video_info, VideoInfo


class PipelineWorker(QThread):
    """后台处理线程 — 5 阶段流水线"""

    # 信号
    progress = Signal(int, str)            # (百分比, 状态描述)
    finished = Signal(dict)                # 完成的剧本 JSON
    error = Signal(str)                    # 错误信息
    subtitle_found = Signal(str, str)      # (字幕来源, 预览文本)

    def __init__(self, video_path: str, parent=None):
        super().__init__(parent)
        self.video_path = video_path
        self._cancelled = False

    def cancel(self):
        """取消处理"""
        self._cancelled = True

    # ================================================================
    # 主流程
    # ================================================================

    def run(self):
        """执行完整流水线"""
        try:
            # ---- 阶段 1: 视频元数据 (5%) ----
            self._check_cancelled()
            self.progress.emit(5, "读取视频信息...")
            info = get_video_info(self.video_path)
            video_title = Path(self.video_path).stem

            # ---- 阶段 2: 字幕提取 (10%–30%) ----
            self._check_cancelled()
            self.progress.emit(10, "提取字幕...")
            subtitle_text, subtitle_source = self._extract_subtitles(info)

            # 如果没有软字幕，启动 Whisper 语音识别
            if subtitle_source not in ("soft",) and info.has_audio_stream:
                self.progress.emit(12, "提取音频 + 语音识别（Whisper）...")
                whisper_text, whisper_source = self._whisper_transcribe()
                if whisper_text:
                    subtitle_text = whisper_text
                    subtitle_source = whisper_source
                    preview = subtitle_text[:200] + "..." if len(subtitle_text) > 200 else subtitle_text
                    self.subtitle_found.emit(subtitle_source, preview)
                    self.progress.emit(28, f"语音识别完成（{len(subtitle_text)} 字）")
                else:
                    self.progress.emit(28, "语音识别失败或无对话")
            elif subtitle_text:
                preview = subtitle_text[:200] + "..." if len(subtitle_text) > 200 else subtitle_text
                self.subtitle_found.emit(subtitle_source, preview)
                self.progress.emit(28, f"字幕已提取（{subtitle_source}）")
            else:
                self.progress.emit(28, "无音频流，将仅依赖视觉分析")

            # ---- 阶段 3: 关键帧提取 (30%–42%) ----
            self._check_cancelled()
            self.progress.emit(30, "检测场景边界...")
            keyframes = self._extract_keyframes(info, subtitle_text)
            self.progress.emit(42, f"提取到 {len(keyframes)} 个关键帧")

            # ---- 阶段 4: GLM-4V 视觉理解 + 硬字幕全帧 OCR (42%–72%) ----
            self._check_cancelled()
            self.progress.emit(42, "AI 分析画面（智谱 GLM-4V）+ 字幕 OCR...")
            scene_descriptions, hard_sub_text = self._analyze_frames(
                keyframes, subtitle_source, len(keyframes)
            )
            self.progress.emit(72, f"完成 {len(scene_descriptions)} 个场景描述")

            # 合并硬字幕 OCR 结果到对话文本
            if hard_sub_text:
                if subtitle_text:
                    subtitle_text = subtitle_text + "\n\n[硬字幕 OCR 补充]\n" + hard_sub_text
                else:
                    subtitle_text = hard_sub_text
                    subtitle_source = "hard_ocr"
                    preview = subtitle_text[:200] + "..." if len(subtitle_text) > 200 else subtitle_text
                    self.subtitle_found.emit(subtitle_source, preview)

            # ---- 术语校正：用 OCR 校正 Whisper 的专有名词 (72%–74%) ----
            if subtitle_source in ("whisper", "whisper+ocr") and hard_sub_text:
                self._check_cancelled()
                self.progress.emit(72, "校正专有名词（OCR 辅助）...")
                subtitle_text = self._correct_terms(
                    dialogue_text=subtitle_text,
                    ocr_text=hard_sub_text,
                )
                if subtitle_text:
                    self.progress.emit(74, f"术语校正完成（{len(subtitle_text)} 字）")
                else:
                    self.progress.emit(74, "术语校正失败，使用原始文本")

            # ---- 阶段 5: DeepSeek 剧本融合 (74%–95%) ----
            self._check_cancelled()
            self.progress.emit(74, "AI 生成剧本（DeepSeek V4 Pro）...")
            script_json = self._generate_script(
                video_title, info, subtitle_text or "",
                subtitle_source, scene_descriptions,
            )
            self.progress.emit(95, "剧本生成完毕")

            # ---- 完成 ----
            self.progress.emit(100, "完成！")
            self.finished.emit(script_json)

        except self._CancelledException:
            self.progress.emit(0, "已取消")
        except Exception as e:
            detail = traceback.format_exc()
            self.error.emit(f"{e}\n\n{detail}")

    # ================================================================
    # 阶段 2: 字幕提取
    # ================================================================

    def _extract_subtitles(self, info: VideoInfo) -> tuple[Optional[str], str]:
        """三级字幕提取"""
        from src.services.audio.subtitle_extractor import extract_subtitles
        return extract_subtitles(self.video_path)

    def _whisper_transcribe(self) -> tuple[Optional[str], str]:
        """使用 Whisper 从音频转录对话"""
        from src.services.audio.subtitle_extractor import extract_subtitles_via_whisper
        return extract_subtitles_via_whisper(self.video_path)

    # ================================================================
    # 阶段 3: 关键帧提取
    # ================================================================

    def _extract_keyframes(self, info: VideoInfo, subtitle_text: Optional[str]) -> list:
        """场景检测 + 关键帧提取 + 质量过滤"""
        from src.services.visual.frame_extractor import FrameExtractor

        # 估算对话占比
        dialogue_ratio = 0.0
        if subtitle_text:
            total_chars = len(subtitle_text)
            estimated_lines = total_chars / 20  # 假设每行 20 字
            dialogue_ratio = min(estimated_lines / max(info.duration_seconds, 1), 1.0)

        extractor = FrameExtractor()
        budget = extractor.calculate_budget(info.duration_seconds, dialogue_ratio)
        keyframes = extractor.extract(self.video_path, info.duration_seconds, budget)

        return keyframes

    # ================================================================
    # 阶段 4: GLM-4V 视觉理解
    # ================================================================

    def _analyze_frames(
        self,
        keyframes: list,
        subtitle_source: str,
        total_frames: int,
    ) -> tuple[list[str], str]:
        """用 GLM-4V 分析关键帧，同时对每帧检测硬字幕"""
        from src.services.fusion.glm_vision import GLMVisionClient

        client = GLMVisionClient()
        # 没有软字幕且没有 Whisper 时，额外做硬字幕 OCR
        need_hard_ocr = subtitle_source not in ("soft", "whisper")

        descriptions: list[str] = []
        hard_sub_lines: list[str] = []
        done = 0

        for kf in keyframes:
            self._check_cancelled()

            # 场景描述（全部用 Plus 模型）
            desc = client.describe_scene(kf.frame, check_subtitle=need_hard_ocr)
            descriptions.append(
                f"[场景{kf.scene_number} | {kf.timestamp_str}] {desc}"
            )

            # 硬字幕检测：每帧都检测（免费 Flash 模型）
            if need_hard_ocr:
                try:
                    has_sub, text = client.detect_hard_subtitle(kf.frame)
                    if has_sub and text:
                        hard_sub_lines.append(f"[{kf.timestamp_str}] {text}")
                except Exception:
                    pass

            done += 1
            # 进度：42% → 72%，按帧数线性插值
            pct = 42 + int(30 * done / max(total_frames, 1))
            self.progress.emit(pct, f"AI 分析画面 ({done}/{total_frames})...")

        return descriptions, "\n".join(hard_sub_lines)

    # ================================================================
    # 术语校正：用 OCR 校正 Whisper 的专有名词
    # ================================================================

    def _correct_terms(self, dialogue_text: str, ocr_text: str) -> str:
        """用 OCR 硬字幕文本作为术语权威来源，校正 Whisper 转录中的专有名词"""
        from src.services.fusion.script_generator import ScriptGenerator

        generator = ScriptGenerator()
        return generator.correct_proper_nouns(dialogue_text, ocr_text)

    # ================================================================
    # 阶段 5: DeepSeek 剧本融合
    # ================================================================

    def _generate_script(
        self,
        title: str,
        info: VideoInfo,
        dialogue_text: str,
        subtitle_source: str,
        scene_descriptions: list[str],
    ) -> dict:
        """用 DeepSeek V4 Pro 融合生成结构化剧本"""
        from src.services.fusion.script_generator import ScriptGenerator

        generator = ScriptGenerator()
        raw_json = generator.generate(
            video_title=title,
            duration_str=info.duration_str,
            characters=[],  # DeepSeek 会从对话中推断
            scene_descriptions=scene_descriptions,
            dialogue_text=dialogue_text,
            subtitle_source=subtitle_source,
        )

        return self._parse_json(raw_json)

    # ================================================================
    # 辅助方法
    # ================================================================

    def _parse_json(self, raw: str) -> dict:
        """解析 DeepSeek 返回的 JSON（处理可能的 markdown 包裹）"""
        raw = raw.strip()
        # 去除可能的 markdown 代码块
        if raw.startswith("```"):
            lines = raw.split("\n")
            # 去掉首行 ```json 和末行 ```
            if lines[-1].strip() == "```":
                lines = lines[1:-1]
            else:
                lines = lines[1:]
            raw = "\n".join(lines)

        return json.loads(raw)

    def _check_cancelled(self):
        """检查是否被取消"""
        if self._cancelled:
            raise self._CancelledException()

    class _CancelledException(Exception):
        """用户取消标记"""
        pass
