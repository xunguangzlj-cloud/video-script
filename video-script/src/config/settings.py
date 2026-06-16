"""配置管理 — 从环境变量和 .env 文件读取 API Key"""

import os
from pathlib import Path
from dotenv import load_dotenv

# 项目根目录
ROOT_DIR = Path(__file__).resolve().parent.parent.parent

# 加载 .env
load_dotenv(ROOT_DIR / ".env")


class Settings:
    """应用配置"""

    # -- API Keys --
    ZHIPU_API_KEY: str = os.getenv("ZHIPU_API_KEY", "")
    DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "")

    # -- 智谱 GLM --
    ZHIPU_BASE_URL: str = "https://open.bigmodel.cn/api/paas/v4/"
    ZHIPU_VISION_MODEL: str = "glm-4v-plus"  # 视觉理解（付费，¥4/百万tokens）
    ZHIPU_VISION_FLASH: str = "glm-4v-flash"  # 视觉理解（免费）

    # -- DeepSeek --
    DEEPSEEK_BASE_URL: str = "https://api.deepseek.com"
    DEEPSEEK_MODEL: str = "deepseek-v4-pro"  # 剧本融合

    # -- 帧处理 --
    FRAME_MAX_EDGE: int = 1024         # 帧缩放最大边长（px）
    FRAME_QUALITY: int = 85            # PNG 压缩质量
    BATCH_SIZE: int = 6                # 每批送入视觉模型的帧数

    # -- 帧数预算 --
    SHORT_VIDEO_MIN: int = 10          # 短片阈值（分钟）
    SHORT_FRAME_BUDGET: int = 30       # 短片最多帧数
    MEDIUM_FRAME_RATE: float = 1.0     # 中等视频帧/分钟
    LONG_FRAME_CAP: int = 80           # 长片帧数上限
    DIALOGUE_HEAVY_RATIO: float = 0.6  # 对话占比阈值（超过则减帧）

    # -- FFmpeg --
    FFMPEG_PATH: str = os.getenv("FFMPEG_PATH", "ffmpeg")
    FFPROBE_PATH: str = os.getenv("FFPROBE_PATH", "ffprobe")

    @classmethod
    def is_ready(cls) -> bool:
        """检查必要 API Key 是否已配置"""
        missing = []
        if not cls.ZHIPU_API_KEY:
            missing.append("智谱 API Key (ZHIPU_API_KEY)")
        if not cls.DEEPSEEK_API_KEY:
            missing.append("DeepSeek API Key (DEEPSEEK_API_KEY)")
        return len(missing) == 0

    @classmethod
    def missing_keys(cls) -> list[str]:
        """返回缺少的 Key 名称列表"""
        missing = []
        if not cls.ZHIPU_API_KEY:
            missing.append("智谱 API Key (ZHIPU_API_KEY)")
        if not cls.DEEPSEEK_API_KEY:
            missing.append("DeepSeek API Key (DEEPSEEK_API_KEY)")
        return missing
