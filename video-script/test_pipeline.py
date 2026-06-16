"""CLI 测试脚本 — 直接运行完整流水线（不需要 GUI）
字幕策略：软字幕 → Whisper 语音识别（主） → 硬字幕 OCR（辅）
"""
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config.settings import Settings
from src.utils.video_info import get_video_info
from src.services.audio.subtitle_extractor import (
    extract_subtitles,
    extract_subtitles_via_whisper,
)
from src.services.visual.frame_extractor import FrameExtractor
from src.services.fusion.glm_vision import GLMVisionClient
from src.services.fusion.script_generator import ScriptGenerator


def run_pipeline(video_path: str):
    print(f"[START] 开始处理视频: {video_path}")
    print(f"   文件大小: {Path(video_path).stat().st_size / 1024 / 1024:.1f} MB")

    # ---- 阶段 1: 视频元数据 ----
    print("\n[阶段 1/6] 读取视频信息...")
    t0 = time.time()
    info = get_video_info(video_path)
    video_title = Path(video_path).stem
    print(f"   时长: {info.duration_str}")
    print(f"   分辨率: {info.width}x{info.height}")
    print(f"   帧率: {info.fps:.1f} fps")
    print(f"   编码: {info.codec}")
    print(f"   字幕流: {'有' if info.has_subtitle_stream else '无'}")
    print(f"   音频流: {'有' if info.has_audio_stream else '无'}")
    print(f"   耗时: {time.time() - t0:.1f}s")

    # ---- 阶段 2: 字幕提取（软字幕 → Whisper） ----
    print("\n[阶段 2/6] 提取字幕...")
    t0 = time.time()
    subtitle_text, subtitle_source = extract_subtitles(video_path)
    print(f"   软字幕: {'有' if subtitle_text else '无'}")

    # 如果没有软字幕，启动 Whisper 语音识别
    if subtitle_source not in ("soft",) and info.has_audio_stream:
        print("   启动 Whisper 语音识别...")
        whisper_text, whisper_source = extract_subtitles_via_whisper(video_path)
        if whisper_text:
            subtitle_text = whisper_text
            subtitle_source = whisper_source
            print(f"   Whisper 完成: {len(subtitle_text)} 字符")
        else:
            print(f"   Whisper 失败或无对话")

    if subtitle_text:
        preview = subtitle_text[:300] + "..." if len(subtitle_text) > 300 else subtitle_text
        print(f"   来源: {subtitle_source}")
        print(f"   长度: {len(subtitle_text)} 字符")
        print(f"   预览:\n{preview[:300]}")
    else:
        print(f"   来源: {subtitle_source}（无字幕，将依赖视觉分析）")
    print(f"   耗时: {time.time() - t0:.1f}s")

    # ---- 阶段 3: 关键帧提取 ----
    print("\n[阶段 3/6] 提取关键帧...")
    t0 = time.time()

    dialogue_ratio = 0.0
    if subtitle_text:
        total_chars = len(subtitle_text)
        estimated_lines = total_chars / 20
        dialogue_ratio = min(estimated_lines / max(info.duration_seconds, 1), 1.0)

    extractor = FrameExtractor()
    budget = extractor.calculate_budget(info.duration_seconds, dialogue_ratio)
    print(f"   时长: {info.duration_minutes:.1f} 分钟")
    print(f"   对话占比: {dialogue_ratio:.0%}")
    print(f"   预算: {budget.total_frames} 帧")

    keyframes = extractor.extract(video_path, info.duration_seconds, budget)
    print(f"   实际提取: {len(keyframes)} 个关键帧")
    for kf in keyframes[:5]:
        print(f"     - [场景{kf.scene_number}] {kf.timestamp_str}")
    if len(keyframes) > 5:
        print(f"     - ... 还有 {len(keyframes) - 5} 帧")
    print(f"   耗时: {time.time() - t0:.1f}s")

    # ---- 阶段 4: GLM-4V 视觉理解 + 硬字幕全帧 OCR ----
    print(f"\n[阶段 4/6] AI 分析画面（智谱 GLM-4V）+ 字幕 OCR...")
    print(f"   共 {len(keyframes)} 帧需要分析")
    t0 = time.time()

    client = GLMVisionClient()
    # 硬字幕 OCR：有 Whisper 时仍做辅助检测（每帧都扫）
    need_hard_ocr = subtitle_source not in ("soft", "whisper")
    ocr_as_supplement = subtitle_source == "whisper"

    descriptions: list[str] = []
    hard_sub_lines: list[str] = []
    ocr_count = 0

    for i, kf in enumerate(keyframes):
        print(f"   分析第 {i+1}/{len(keyframes)} 帧 [场景{kf.scene_number}] {kf.timestamp_str}...", end=" ", flush=True)
        t_frame = time.time()

        try:
            desc = client.describe_scene(kf.frame, check_subtitle=need_hard_ocr)
            descriptions.append(
                f"[场景{kf.scene_number} | {kf.timestamp_str}] {desc}"
            )
            print(f"OK ({time.time() - t_frame:.1f}s) {desc[:80]}...")
        except Exception as e:
            print(f"FAIL: {e}")
            descriptions.append(f"[场景{kf.scene_number} | {kf.timestamp_str}] (分析失败)")

        # 硬字幕 OCR：每帧都检测（免费 Flash 模型）
        if need_hard_ocr or ocr_as_supplement:
            try:
                has_sub, text = client.detect_hard_subtitle(kf.frame)
                if has_sub and text:
                    hard_sub_lines.append(f"[{kf.timestamp_str}] {text}")
                    ocr_count += 1
                    print(f"     [OCR] {text[:60]}...")
            except Exception:
                pass

    print(f"   场景描述: {len(descriptions)} 条")
    print(f"   硬字幕 OCR: {ocr_count} 处检出")
    print(f"   耗时: {time.time() - t0:.1f}s")

    # ---- 合并字幕 ----
    if hard_sub_lines:
        hard_text = "\n".join(hard_sub_lines)
        if subtitle_text:
            subtitle_text = subtitle_text + "\n\n[硬字幕 OCR 补充]\n" + hard_text
            if ocr_as_supplement:
                subtitle_source = "whisper+ocr"
        else:
            subtitle_text = hard_text
            subtitle_source = "hard_ocr"

    # ---- 术语校正：用 OCR 校正 Whisper 的专有名词 ----
    if subtitle_source in ("whisper", "whisper+ocr") and hard_sub_lines:
        print(f"\n[术语校正] 用 OCR 硬字幕校正 Whisper 专有名词...")
        t_correct = time.time()
        try:
            temp_generator = ScriptGenerator()
            corrected = temp_generator.correct_proper_nouns(
                dialogue_text=subtitle_text,
                ocr_text=hard_text,
            )
            if corrected:
                print(f"   校正前: {len(subtitle_text)} 字")
                print(f"   校正后: {len(corrected)} 字")
                subtitle_text = corrected
                print(f"   耗时: {time.time() - t_correct:.1f}s")
            else:
                print(f"   校正失败，使用原始文本")
        except Exception as e:
            print(f"   校正出错: {e}，使用原始文本")

    # ---- 阶段 5: DeepSeek 剧本融合 ----
    print(f"\n[阶段 5/6] AI 生成剧本（DeepSeek V4 Pro）...")
    t0 = time.time()

    generator = ScriptGenerator()

    print(f"   场景描述: {len(descriptions)} 条")
    print(f"   对话文本: {len(subtitle_text or '')} 字符")
    print(f"   字幕来源: {subtitle_source}")
    print(f"   调用 DeepSeek...")

    try:
        raw_json = generator.generate(
            video_title=video_title,
            duration_str=info.duration_str,
            characters=[],
            scene_descriptions=descriptions,
            dialogue_text=subtitle_text or "",
            subtitle_source=subtitle_source,
        )

        print(f"   耗时: {time.time() - t0:.1f}s")

        # 解析 JSON
        raw_json = raw_json.strip()
        if raw_json.startswith("```"):
            lines = raw_json.split("\n")
            if lines[-1].strip() == "```":
                lines = lines[1:-1]
            else:
                lines = lines[1:]
            raw_json = "\n".join(lines)

        script = json.loads(raw_json)

        # 保存结果
        output_path = Path(video_path).with_suffix(".json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(script, f, ensure_ascii=False, indent=2)

        print(f"\n[DONE] 剧本已保存到: {output_path}")

        # 打印摘要
        meta = script.get("metadata", {})
        chars = script.get("characters", [])
        scenes = script.get("scenes", [])
        dialogue_count = sum(len(sc.get("dialogues", [])) for sc in scenes)
        print(f"\n[SUMMARY] 剧本摘要:")
        print(f"   标题: {meta.get('title', 'N/A')[:80]}...")
        print(f"   时长: {meta.get('duration', 'N/A')}")
        print(f"   字幕来源: {meta.get('subtitle_source', 'N/A')}")
        print(f"   角色数: {len(chars)}")
        print(f"   场景数: {len(scenes)}")
        print(f"   台词总数: {dialogue_count}")
        for c in chars:
            print(f"     - {c.get('id', '?')}: {c.get('name', '?')}")

        return script

    except Exception as e:
        print(f"\n[ERROR] DeepSeek 调用失败: {e}")
        traceback.print_exc()
        return None


if __name__ == "__main__":
    video = sys.argv[1] if len(sys.argv) > 1 else None
    if not video:
        video = r"D:\视频\鸣潮剧情合集三期4K（更新到3.4赛博朋克2077，边缘行者联动剧情，驶向尚未点亮之星，洛瑟菈我们选择的天空） 【3.1.主】驶向尚未点亮之星3远航星1.mp4"

    if not Path(video).exists():
        print(f"[ERROR] 视频文件不存在: {video}")
        sys.exit(1)

    print(f"Settings:")
    print(f"  FFmpeg: {Settings.FFMPEG_PATH}")
    print(f"  FFprobe: {Settings.FFPROBE_PATH}")
    print(f"  智谱 Model: {Settings.ZHIPU_VISION_MODEL}")
    print(f"  DeepSeek Model: {Settings.DEEPSEEK_MODEL}")
    print(f"  Keys Ready: {Settings.is_ready()}")
    print()

    run_pipeline(video)
