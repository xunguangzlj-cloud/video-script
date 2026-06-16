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

    def correct_proper_nouns(
        self,
        dialogue_text: str,
        ocr_text: str,
    ) -> str:
        """用 OCR 硬字幕文本作为术语权威来源，校正 Whisper 转录中的专有名词

        Whisper 是"听音辨字"，对游戏/影视专有名词（人名、地名、术语）容易出错；
        硬字幕 OCR 从画面中直接提取文字，专有名词准确率高。
        本方法用 DeepSeek 将两者对齐，输出校正后的对话文本。

        Args:
            dialogue_text: Whisper 转录的原始对话文本
            ocr_text: 硬字幕 OCR 提取的文本（专有名词正确）

        Returns:
            校正后的对话文本
        """
        prompt = (
            "## 任务\n"
            "你是一位文字校对专家。下面有两份来自同一视频的文本：\n\n"
            "### A. 语音转录文本（Whisper）\n"
            "由语音识别生成，**对话覆盖全面**，但**专有名词可能听错**"
            "（人名、地名、术语等因同音/近音被误识别）。\n\n"
            "### B. 硬字幕 OCR 文本\n"
            "从视频画面的字幕中直接提取，**专有名词准确**，但只覆盖了部分对话。\n\n"
            "## 要求\n"
            "1. 以 A 为主体，保持其完整的对话覆盖和时间戳\n"
            "2. 用 B 中的**正确专有名词**替换 A 中对应的**错误/近似写法**"
            "（人名、地名、组织名、特殊术语等）\n"
            "3. 不要改动 A 中与专有名词无关的普通对话内容\n"
            "4. 如果 B 中出现了 A 完全没有的台词，在对应时间位置补充进去\n"
            "5. 直接输出校正后的完整文本，不要加任何解释说明\n\n"
            f"### A. 语音转录文本\n{dialogue_text[:400000]}\n\n"
            f"### B. 硬字幕 OCR 文本\n{ocr_text[:100000]}"
        )

        response = self.client.chat.completions.create(
            model=Settings.DEEPSEEK_MODEL,
            messages=[
                {"role": "user", "content": prompt},
            ],
            max_tokens=200000,
            temperature=0.2,  # 低温度，校正任务需要精确
        )

        return response.choices[0].message.content

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
