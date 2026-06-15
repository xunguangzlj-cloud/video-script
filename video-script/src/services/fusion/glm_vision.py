"""智谱 GLM-4V 视觉理解 — 帧 → 场景描述"""

import base64
import io
from typing import Optional

import cv2
import numpy as np
from openai import OpenAI

from src.config.settings import Settings


class GLMVisionClient:
    """智谱 GLM-4V 视觉理解客户端"""

    def __init__(self):
        self.client = OpenAI(
            api_key=Settings.ZHIPU_API_KEY,
            base_url=Settings.ZHIPU_BASE_URL,
        )
        self.model = Settings.ZHIPU_VISION_MODEL

    def encode_frame(self, frame: np.ndarray) -> str:
        """将 numpy 帧编码为 base64 PNG"""
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        _, buffer = cv2.imencode(
            ".png",
            frame,
            [cv2.IMWRITE_PNG_COMPRESSION, 3],
        )
        return base64.b64encode(buffer).decode("utf-8")

    def resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """缩放帧到最大边长 1024px"""
        h, w = frame.shape[:2]
        max_edge = Settings.FRAME_MAX_EDGE
        if max(h, w) > max_edge:
            scale = max_edge / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            frame = cv2.resize(frame, (new_w, new_h))
        return frame

    def describe_scene(self, frame: np.ndarray, check_subtitle: bool = False) -> str:
        """描述单个场景帧

        Args:
            frame: OpenCV BGR 帧
            check_subtitle: 是否同时做 OCR（用于检测硬字幕）

        Returns:
            场景描述文本
        """
        frame = self.resize_frame(frame)
        b64_data = self.encode_frame(frame)

        prompt = self._get_scene_prompt(check_subtitle)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64_data}"},
                    },
                    {"type": "text", "text": prompt},
                ],
            }],
            max_tokens=512,
            temperature=0.3,
        )

        return response.choices[0].message.content.strip()

    def describe_batch(
        self,
        frames: list[np.ndarray],
        check_subtitles: bool = False,
    ) -> list[str]:
        """批量描述场景帧（逐个调用，每帧最多5张图片）"""
        results = []
        batch_size = Settings.BATCH_SIZE

        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i : i + batch_size]
            batch_b64 = [self.encode_frame(self.resize_frame(f)) for f in batch_frames]

            # 构建多图消息
            content = []
            for b64 in batch_b64:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                })
            content.append({
                "type": "text",
                "text": self._get_batch_prompt(len(batch_frames), check_subtitles),
            })

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": content}],
                max_tokens=2048,
                temperature=0.3,
            )

            batch_text = response.choices[0].message.content.strip()
            # 按帧拆分描述
            descriptions = self._split_batch_result(batch_text, batch_frames)
            results.extend(descriptions)

        return results

    def detect_hard_subtitle(self, frame: np.ndarray) -> tuple[bool, str]:
        """检测帧中是否有硬字幕，并返回 OCR 文本

        Returns:
            (是否有字幕, OCR文本)
        """
        frame = self.resize_frame(frame)
        b64_data = self.encode_frame(frame)

        response = self.client.chat.completions.create(
            model=Settings.ZHIPU_VISION_FLASH,  # 用免费版检测
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64_data}"},
                    },
                    {
                        "type": "text",
                        "text": "这张图片底部是否有字幕文字？如果有，请逐行输出所有字幕文本。如果没有任何字幕，只回复'无字幕'。",
                    },
                ],
            }],
            max_tokens=256,
        )

        text = response.choices[0].message.content.strip()
        has_subtitle = "无字幕" not in text
        return has_subtitle, text if has_subtitle else ""

    def _get_scene_prompt(self, check_subtitle: bool) -> str:
        """场景描述 Prompt"""
        base = (
            "请用一段中文描述这个画面，包含以下信息：\n"
            "1. 场景类型（内景/外景、地点）\n"
            "2. 出场人物（数量、特征、动作）\n"
            "3. 光线和色调\n"
            "4. 镜头类型（远景/中景/特写等）\n"
            "5. 画面氛围和情绪\n"
        )
        if check_subtitle:
            base += "6. 画面中的字幕文本（如有）\n"
        return base

    def _get_batch_prompt(self, frame_count: int, check_subtitles: bool) -> str:
        """批量场景描述 Prompt"""
        prompt = (
            f"以下是按时间顺序排列的 {frame_count} 个视频帧。"
            "请为每帧提供一段中文场景描述，用'---'分隔每帧。\n"
            "每帧描述需包含：地点、人物、动作、光线、镜头类型、氛围\n"
        )
        if check_subtitles:
            prompt += "如果帧中有字幕文字，请在描述末尾注明'[字幕: xxx]'。\n"
        return prompt

    def _split_batch_result(self, text: str, frames: list) -> list[str]:
        """按分隔符拆分批量结果"""
        parts = [p.strip() for p in text.split("---") if p.strip()]
        # 确保帧数匹配
        while len(parts) < len(frames):
            parts.append("")
        return parts[: len(frames)]
