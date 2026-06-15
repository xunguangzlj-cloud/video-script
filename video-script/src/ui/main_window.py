"""主窗口 — PySide6 桌面界面"""

import sys
from pathlib import Path

from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenuBar,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QStatusBar,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtCore import Qt, QThread, Signal

from src.config.settings import Settings
from src.utils.video_info import get_video_info
from src.utils.ffmpeg_helper import check_ffmpeg, check_ffprobe


class MainWindow(QMainWindow):
    """视频转剧本 - 主窗口"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("视频转文字剧本")
        self.video_path: str | None = None

        self._check_environment()
        self._build_ui()
        self._build_menu()
        self._check_api_keys()

    # ---------- 环境检查 ----------

    def _check_environment(self):
        """检查 FFmpeg 等依赖"""
        if not check_ffmpeg():
            QMessageBox.warning(
                self, "缺少依赖",
                "未检测到 FFmpeg，请先安装 FFmpeg 并添加到 PATH。\n"
                "下载地址: https://ffmpeg.org/download.html",
            )

    def _check_api_keys(self):
        """检查 API Key 配置状态"""
        missing = Settings.missing_keys()
        if missing:
            self.statusBar().showMessage(
                f"⚠ 缺少 API Key: {', '.join(missing)} — 请编辑 .env 文件"
            )
        else:
            self.statusBar().showMessage("✅ API Key 已就绪，可以开始处理")

    # ---------- 菜单栏 ----------

    def _build_menu(self):
        menu_bar = self.menuBar()

        # 设置菜单
        settings_menu = menu_bar.addMenu("设置(&S)")
        settings_menu.addAction("打开 .env 文件", self._open_env_file)

        # 帮助菜单
        help_menu = menu_bar.addMenu("帮助(&H)")
        help_menu.addAction("关于", self._show_about)

    # ---------- UI 构建 ----------

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setSpacing(12)

        # ---- 导入区域 ----
        import_layout = QHBoxLayout()
        self.btn_import = QPushButton("📁 选择视频文件")
        self.btn_import.clicked.connect(self._on_import_video)
        self.lbl_file = QLabel("未选择文件")
        self.lbl_file.setStyleSheet("color: #888;")
        import_layout.addWidget(self.btn_import)
        import_layout.addWidget(self.lbl_file, 1)
        layout.addLayout(import_layout)

        # ---- 视频信息 ----
        self.lbl_info = QLabel("")
        self.lbl_info.setStyleSheet("padding: 4px;")
        layout.addWidget(self.lbl_info)

        # ---- 操作按钮 ----
        btn_layout = QHBoxLayout()
        self.btn_start = QPushButton("▶ 开始生成剧本")
        self.btn_start.setEnabled(False)
        self.btn_start.setMinimumHeight(40)
        self.btn_start.setStyleSheet(
            "QPushButton { background-color: #4A90D9; color: white; font-size: 14px; border-radius: 6px; }"
            "QPushButton:hover { background-color: #357ABD; }"
            "QPushButton:disabled { background-color: #ccc; }"
        )
        self.btn_start.clicked.connect(self._on_start)

        self.btn_cancel = QPushButton("取消")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self._on_cancel)

        btn_layout.addWidget(self.btn_start)
        btn_layout.addWidget(self.btn_cancel)
        layout.addLayout(btn_layout)

        # ---- 进度条 ----
        progress_layout = QHBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.lbl_status = QLabel("")
        progress_layout.addWidget(self.progress_bar, 1)
        progress_layout.addWidget(self.lbl_status)
        layout.addLayout(progress_layout)

        # ---- 剧本查看器 ----
        self.script_viewer = QTextEdit()
        self.script_viewer.setReadOnly(True)
        self.script_viewer.setPlaceholderText(
            "生成的剧本将在此显示...\n\n"
            "使用方法：\n"
            "1. 点击「选择视频文件」导入视频\n"
            "2. 点击「开始生成剧本」开始处理\n"
            "3. 等待完成后保存或导出"
        )
        layout.addWidget(self.script_viewer, 1)

        # ---- 导出按钮 ----
        export_layout = QHBoxLayout()
        self.btn_export = QPushButton("💾 导出 JSON")
        self.btn_export.setEnabled(False)
        export_layout.addWidget(self.btn_export)
        export_layout.addStretch()
        layout.addLayout(export_layout)

    # ---------- 槽函数 ----------

    def _on_import_video(self):
        """选择视频文件"""
        path, _ = QFileDialog.getOpenFileName(
            self, "选择视频文件", "",
            "视频文件 (*.mp4 *.mkv *.avi *.mov *.flv *.ts);;所有文件 (*.*)"
        )
        if not path:
            return

        self.video_path = path
        self.lbl_file.setText(path)
        self.lbl_file.setStyleSheet("color: #333;")

        # 获取视频元数据
        info = get_video_info(path)
        self.lbl_info.setText(
            f"时长: {info.duration_str}  |  "
            f"分辨率: {info.width}x{info.height}  |  "
            f"编码: {info.codec}  |  "
            f"帧率: {info.fps:.1f} fps  |  "
            f"字幕流: {'有' if info.has_subtitle_stream else '无'}  |  "
            f"字幕类型: {info.subtitle_type}"
        )

        self.btn_start.setEnabled(True)

    def _on_start(self):
        """开始处理"""
        if not self.video_path:
            return

        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.btn_start.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.btn_import.setEnabled(False)
        self.lbl_status.setText("正在处理...")

        # TODO: 启动流水线（后续阶段实现）
        self.script_viewer.setPlainText("处理中...\n（流水线将在后续阶段实现）")

    def _on_cancel(self):
        """取消处理"""
        self.btn_cancel.setEnabled(False)
        self.btn_start.setEnabled(True)
        self.btn_import.setEnabled(True)
        self.lbl_status.setText("已取消")
        self.progress_bar.setVisible(False)

    def _open_env_file(self):
        """打开 .env 配置文件"""
        env_path = Path(__file__).resolve().parent.parent.parent / ".env"
        if sys.platform == "win32":
            import os
            os.startfile(str(env_path))
        else:
            import subprocess
            subprocess.run(["open" if sys.platform == "darwin" else "xdg-open", str(env_path)])

    def _show_about(self):
        QMessageBox.about(
            self, "关于",
            "视频转文字剧本 v1.0\n\n"
            "将剧情视频自动转换为结构化文字剧本。\n\n"
            "技术栈：PySide6 + 智谱 GLM-4V + DeepSeek V4 Pro\n\n"
            "AI 模型：\n"
            "  - 视觉理解：智谱 GLM-4V-Plus\n"
            "  - 剧本融合：DeepSeek V4 Pro\n"
            "  - 语音识别：faster-whisper（兜底）\n\n"
            "成本：约 ¥0.27/部电影"
        )
