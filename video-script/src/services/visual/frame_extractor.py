"""关键帧提取 — PySceneDetect 场景检测 + OpenCV 质量过滤"""

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import scenedetect
from scenedetect import AdaptiveDetector


@dataclass
class KeyFrame:
    """关键帧"""
    timestamp_sec: float
    timestamp_str: str          # HH:MM:SS
    frame: np.ndarray            # BGR 图像
    scene_number: int
    is_dialogue_heavy: bool = False


@dataclass
class FrameBudget:
    """帧数预算"""
    total_frames: int
    max_per_minute: float
    total_cap: int


class FrameExtractor:
    """关键帧提取器：场景检测 + 质量过滤 + 预算控制"""

    def __init__(
        self,
        adaptive_threshold: float = 2.0,
        min_scene_len: int = 8,
        min_brightness: float = 20.0,
        min_sharpness: float = 100.0,
        max_similarity: float = 0.92,
    ):
        """
        Args:
            adaptive_threshold: 自适应检测阈值（越低越敏感）
            min_scene_len: 最短场景长度（帧数）
            min_brightness: 最低平均亮度（0-255）
            min_sharpness: 最低锐度（Laplacian方差）
            max_similarity: 与上一关键帧最大相似度（超过则跳过）
        """
        self.adaptive_threshold = adaptive_threshold
        self.min_scene_len = min_scene_len
        self.min_brightness = min_brightness
        self.min_sharpness = min_sharpness
        self.max_similarity = max_similarity
        self._last_keyframe: Optional[np.ndarray] = None

    # ----------------------------------------------------------------
    # 公开方法
    # ----------------------------------------------------------------

    def calculate_budget(self, duration_sec: float, dialogue_ratio: float = 0.0) -> FrameBudget:
        """根据视频时长和对话占比计算帧数预算

        规则（来自 plan）：
        - < 10 分钟：最多 30 帧（3帧/分钟）
        - 10-60 分钟：约 1帧/分钟
        - > 60 分钟：最多 80 帧（0.5帧/分钟）
        - 对话占比高（>60%）：额外减帧 20%
        """
        from src.config.settings import Settings

        duration_min = duration_sec / 60.0

        if duration_min < Settings.SHORT_VIDEO_MIN:
            budget = min(int(duration_min * 3), Settings.SHORT_FRAME_BUDGET)
            rate = 3.0
        elif duration_min <= 60:
            budget = int(duration_min * Settings.MEDIUM_FRAME_RATE)
            rate = Settings.MEDIUM_FRAME_RATE
        else:
            budget = min(int(duration_min * 0.5), Settings.LONG_FRAME_CAP)
            rate = 0.5

        # 对话占比高则减帧
        if dialogue_ratio > Settings.DIALOGUE_HEAVY_RATIO:
            budget = int(budget * 0.8)

        return FrameBudget(
            total_frames=max(budget, 5),
            max_per_minute=rate,
            total_cap=Settings.LONG_FRAME_CAP,
        )

    def extract(self, video_path: str, duration_sec: float, budget: FrameBudget) -> list[KeyFrame]:
        """从视频提取关键帧

        Args:
            video_path: 视频文件路径
            duration_sec: 视频时长（秒）
            budget: 帧数预算

        Returns:
            按时间排序的关键帧列表
        """
        # 安全上限：防止时长为 0 时无限提取
        max_frames = max(budget.total_frames, 1)
        # 绝对安全帽：任何情况下不超过 200 帧，不做无节制的 API 调用
        safety_cap = min(200, max(30, max_frames))

        video = scenedetect.open_video(video_path)
        detector = AdaptiveDetector(
            adaptive_threshold=self.adaptive_threshold,
            min_scene_len=self.min_scene_len,
        )

        # 场景检测（PySceneDetect 0.7 API）
        scene_list = scenedetect.detect(video_path, detector)
        scene_boundaries = [
            (s[0].get_seconds(), s[1].get_seconds()) for s in scene_list
        ]

        if not scene_boundaries:
            # 回退：均匀采样
            return self._fallback_uniform(video_path, duration_sec, budget)

        # 按预算分配帧数，但不超过安全上限
        frames_per_scene = self._allocate_per_scene(scene_boundaries, budget.total_frames)
        # 检查总数是否超过安全上限
        total_alloc = sum(frames_per_scene)
        if total_alloc > safety_cap:
            scale = safety_cap / total_alloc
            frames_per_scene = [max(1, int(f * scale)) for f in frames_per_scene]

        # 提取关键帧
        cap = cv2.VideoCapture(video_path)
        keyframes: list[KeyFrame] = []
        self._last_keyframe = None

        for scene_idx, ((start, end), alloc) in enumerate(
            zip(scene_boundaries, frames_per_scene), start=1
        ):
            scene_frames = self._extract_scene_frames(
                cap, scene_idx, start, end, alloc
            )
            keyframes.extend(scene_frames)

        cap.release()
        return keyframes

    # ----------------------------------------------------------------
    # 帧分配
    # ----------------------------------------------------------------

    def _allocate_per_scene(
        self, scene_boundaries: list[tuple[float, float]], total_budget: int
    ) -> list[int]:
        """按场景时长比例分配帧数

        当场景数超预算时，优先分配给时长最长的场景。
        """
        n_scenes = len(scene_boundaries)
        durations = [end - start for start, end in scene_boundaries]
        total_dur = sum(durations)

        if total_dur <= 0 or n_scenes == 0:
            return [1] * n_scenes

        # 场景数不超过预算：每个场景至少 1 帧，余下按时长比例分配
        if n_scenes <= total_budget:
            allocated = [1] * n_scenes
            remaining = total_budget - n_scenes

            if remaining > 0:
                for i, dur in enumerate(durations):
                    extra = int(remaining * dur / total_dur)
                    allocated[i] += extra

                # 补回舍入损失
                while sum(allocated) < total_budget:
                    idx = max(range(n_scenes), key=lambda i: durations[i] / allocated[i])
                    allocated[idx] += 1

            return allocated

        # 场景数超预算：只给时长最长的 top-N 场景各 1 帧
        indexed = sorted(enumerate(durations), key=lambda x: x[1], reverse=True)
        allocated = [0] * n_scenes
        for rank, (idx, dur) in enumerate(indexed):
            if rank < total_budget:
                allocated[idx] = 1
            else:
                break

        return allocated

    # ----------------------------------------------------------------
    # 场景帧提取
    # ----------------------------------------------------------------

    def _extract_scene_frames(
        self,
        cap: cv2.VideoCapture,
        scene_number: int,
        start_sec: float,
        end_sec: float,
        allocation: int,
    ) -> list[KeyFrame]:
        """从一个场景中提取帧"""
        frames: list[KeyFrame] = []
        duration = end_sec - start_sec

        if allocation <= 0 or duration <= 0:
            return frames

        if allocation == 1:
            # 取场景中间帧
            mid = start_sec + duration / 2
            frame = self._read_frame_at(cap, mid)
            if frame is not None and self._passes_quality(frame):
                frames.append(self._make_keyframe(frame, mid, scene_number))
        else:
            # 均匀采样
            interval = duration / allocation
            for i in range(allocation):
                ts = start_sec + interval * (i + 0.5)
                frame = self._read_frame_at(cap, ts)
                if frame is None:
                    continue
                if not self._passes_quality(frame):
                    continue
                if self._too_similar(frame):
                    continue
                frames.append(self._make_keyframe(frame, ts, scene_number))
                self._last_keyframe = frame.copy()

        return frames

    # ----------------------------------------------------------------
    # 帧读取
    # ----------------------------------------------------------------

    def _read_frame_at(self, cap: cv2.VideoCapture, sec: float) -> Optional[np.ndarray]:
        """读取指定时间点的帧"""
        cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
        ret, frame = cap.read()
        if not ret:
            return None
        return frame

    # ----------------------------------------------------------------
    # 质量过滤
    # ----------------------------------------------------------------

    def _passes_quality(self, frame: np.ndarray) -> bool:
        """检查帧是否通过质量过滤"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 亮度检查
        brightness = gray.mean()
        if brightness < self.min_brightness:
            return False

        # 锐度检查（Laplacian 方差）
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var < self.min_sharpness:
            return False

        return True

    def _too_similar(self, frame: np.ndarray) -> bool:
        """检查与上一关键帧是否过于相似"""
        if self._last_keyframe is None:
            return False

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        last_gray = cv2.cvtColor(self._last_keyframe, cv2.COLOR_BGR2GRAY)

        # 使用归一化相关系数（比 SSIM 快）
        result = cv2.matchTemplate(gray, last_gray, cv2.TM_CCOEFF_NORMED)
        similarity = (result[0][0] + 1) / 2  # 映射到 [0, 1]

        return similarity > self.max_similarity

    # ----------------------------------------------------------------
    # 辅助方法
    # ----------------------------------------------------------------

    @staticmethod
    def _make_keyframe(frame: np.ndarray, ts: float, scene_no: int) -> KeyFrame:
        """构建 KeyFrame 对象"""
        h, m = divmod(int(ts), 3600)
        m, s = divmod(m, 60)
        return KeyFrame(
            timestamp_sec=ts,
            timestamp_str=f"{h:02d}:{m:02d}:{s:02d}",
            frame=frame,
            scene_number=scene_no,
        )

    def _fallback_uniform(
        self, video_path: str, duration_sec: float, budget: FrameBudget
    ) -> list[KeyFrame]:
        """场景检测失败时回退为均匀采样"""
        cap = cv2.VideoCapture(video_path)
        frames: list[KeyFrame] = []
        self._last_keyframe = None

        interval = duration_sec / budget.total_frames
        for i in range(budget.total_frames):
            ts = interval * (i + 0.5)
            frame = self._read_frame_at(cap, ts)
            if frame is None:
                continue
            if not self._passes_quality(frame):
                continue
            frames.append(self._make_keyframe(frame, ts, i + 1))
            self._last_keyframe = frame.copy()

        cap.release()
        return frames
