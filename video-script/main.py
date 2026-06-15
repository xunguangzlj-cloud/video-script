"""视频转文字剧本工具 - 主入口"""

import sys
from pathlib import Path

# 将项目根目录加入路径
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from dotenv import load_dotenv

# 加载 .env 配置
load_dotenv(ROOT_DIR / ".env")


def main():
    from PySide6.QtWidgets import QApplication
    from src.ui.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("视频转剧本")

    window = MainWindow()
    window.resize(1200, 800)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
