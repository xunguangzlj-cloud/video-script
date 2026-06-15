"""DeepSeek V4 Pro 剧本融合生成"""

from openai import OpenAI

from src.config.settings import Settings


class ScriptGenerator:
    """DeepSeek V4 Pro 剧本融合生成器"""

    def __init__(self):
        self.client = OpenAI(
            api_key=Settings.DEEPSEEK_API_KEY,
            base_url=Settings.DEEPSEEK_BASE_URL,
        )

    def generate(
        self,
        video_title: str,
        duration_str: str,
        characters: list[dict],
        scene_descriptions: list[str],
        dialogue_text: str,
        subtitle_source: str,
    ) -> str:
        """融合视觉描述 + 对话文本 → 结构化剧本

        Args:
            video_title: 视频标题
            duration_str: 时长字符串
            characters: 角色列表
            scene_descriptions: 场景描述列表
            dialogue_text: 对话/字幕文本
            subtitle_source: 字幕来源（soft/hard_ocr/whisper）

        Returns:
            JSON 格式的完整剧本
        """
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(
            video_title, duration_str, characters,
            scene_descriptions, dialogue_text, subtitle_source,
        )

        response = self.client.chat.completions.create(
            model=Settings.DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=200000,
            temperature=0.5,
        )

        return response.choices[0].message.content

    def _build_system_prompt(self) -> str:
        return (
            "你是一位资深的影视剧本分析师。你的任务是根据视频的视觉场景描述"
            "和对话/字幕文本，生成一份完整的结构化剧本。\n\n"
            "输出必须为合法的 JSON 格式，结构如下：\n"
            "{\n"
            '  "metadata": {"title": "", "duration": "", "subtitle_source": ""},\n'
            '  "characters": [{"id": "CHAR_01", "name": "", "description": ""}],\n'
            '  "scenes": [{\n'
            '    "scene_number": 1,\n'
            '    "scene_heading": {"location_type": "INT./EXT.", "location": "", "time_of_day": ""},\n'
            '    "visual_description": {"setting": "", "atmosphere": "", "lighting": "", "color_palette": ""},\n'
            '    "actions": [{"character_id": "", "description": ""}],\n'
            '    "dialogues": [{"character_id": "", "line": "", "delivery_notes": ""}],\n'
            '    "scene_notes": ""\n'
            "  }]\n"
            "}\n\n"
            "要求：\n"
            "- 对话文本按时间戳与对应场景匹配\n"
            "- 保持角色名称一致性\n"
            "- 场景编号连续\n"
            "- 使用中文输出"
        )

    def _build_user_prompt(
        self,
        title: str,
        duration: str,
        characters: list[dict],
        scenes: list[str],
        dialogue: str,
        source: str,
    ) -> str:
        return (
            f"## 视频信息\n"
            f"- 标题：{title}\n"
            f"- 时长：{duration}\n"
            f"- 字幕来源：{source}\n"
            f"- 已知角色：{characters}\n\n"
            f"## 场景描述（视觉分析结果）\n"
            f"{chr(10).join(f'- {s}' for s in scenes)}\n\n"
            f"## 对话/字幕文本\n"
            f"{dialogue[:500000]}\n\n"  # 截断以防超长
            f"请根据以上信息，生成完整的结构化剧本 JSON。"
        )
