"""Prompt 模板 — 所有 AI 调用的 Prompt 集中管理"""

# ----------------------------------------------------------------
# GLM-4V 视觉理解
# ----------------------------------------------------------------

SCENE_DESCRIPTION = (
    "请用一段中文描述这个画面，包含以下信息：\n"
    "1. 场景类型（内景/外景、具体地点）\n"
    "2. 出场人物（数量、外观特征、当前动作）\n"
    "3. 光线条件（明亮/昏暗/自然光/人工光）和色调\n"
    "4. 镜头类型（远景/全景/中景/近景/特写）与构图\n"
    "5. 画面传达的氛围和情绪\n"
    "6. 如果画面中有文字（字幕、标牌等），请如实记录\n"
)

SCENE_BATCH = (
    "以下是按时间顺序排列的 {count} 个视频帧。"
    "请为每一帧提供中文场景描述，帧与帧之间用'---'分隔。\n"
    "每帧描述包含：地点、人物、动作、光线、镜头、氛围、画面文字。\n"
)

HARD_SUBTITLE_CHECK = (
    "这张图片底部是否有字幕文字？如果有，请逐行输出所有字幕文本。"
    "如果没有任何字幕，只回复'无字幕'。"
)

# ----------------------------------------------------------------
# DeepSeek 剧本融合
# ----------------------------------------------------------------

SCRIPT_SYSTEM = (
    "你是一位资深的影视剧本分析师。你的任务是根据视频的视觉场景描述"
    "和对话/字幕文本，生成一份完整的结构化剧本。\n\n"
    "输出必须为合法的 JSON 格式：\n"
    "{\n"
    '  "metadata": {"title": "", "duration": "", "subtitle_source": "", "total_scenes": 0},\n'
    '  "characters": [\n'
    '    {"id": "CHAR_01", "name": "角色名", "description": "角色描述", "first_appearance_scene": 1}\n'
    "  ],\n"
    '  "scenes": [\n'
    "    {\n"
    '      "scene_number": 1,\n'
    '      "scene_heading": {"location_type": "INT.", "location": "地点", "time_of_day": "NIGHT"},\n'
    '      "time_range": {"start": "00:00:00", "end": "00:00:30"},\n'
    '      "visual_description": {\n'
    '        "setting": "场景布置描述",\n'
    '        "atmosphere": "氛围",\n'
    '        "lighting": "光线描述",\n'
    '        "color_palette": "色调"\n'
    "      },\n"
    '      "actions": [{"timestamp": "时间", "character_id": "CHAR_01", "description": "动作描述"}],\n'
    '      "dialogues": [{"timestamp": "时间", "character_id": "CHAR_01", "line": "台词", "delivery_notes": "语气"}],\n'
    '      "transitions": {"in": "", "out": ""},\n'
    '      "scene_notes": ""\n'
    "    }\n"
    "  ]\n"
    "}\n\n"
    "规则：\n"
    "1. 对话按时间戳与对应场景匹配\n"
    "2. 角色名称全程保持一致，同一角色不同场景用同一 ID\n"
    "3. 场景编号从 1 开始连续递增\n"
    "4. scene_heading.location_type 只能是 INT./EXT./INT./EXT.\n"
    "5. time_of_day 只能是 DAY/NIGHT/DAWN/DUSK/UNKNOWN\n"
    "6. 所有文本用中文"
)

# ----------------------------------------------------------------
# DeepSeek 角色名规范化
# ----------------------------------------------------------------

CHARACTER_NORMALIZE = (
    "你是一个角色名称分析器。以下是视频中的对话（匿名化说话人为 SPEAKER_A、SPEAKER_B 等）。"
    "请根据上下文推断每个 SPEAKER 的合理角色名，并返回映射 JSON：\n"
    '{"SPEAKER_A": "角色名", "SPEAKER_B": "角色名", ...}\n'
    "如果无法从上下文推断，保持原 SPEAKER 标签。"
)
