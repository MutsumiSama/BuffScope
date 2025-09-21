# app/main.py
# -*- coding: utf-8 -*-
import os
import sys
import cv2

from .utils import (
    imread_u, imwrite_u, load_json, save_json,
    load_anchor_cache, save_anchor_cache
)
from .detector import (
    match_anchor, recognize_in_box,
    compute_row_area_from_anchor, load_gallery,
    enumerate_slots_in_box, recognize_countdowns_in_slots
)
from .visualizer import annotate_and_save
from .cd_manager import compute_and_update_cd

# 路径常量
CFG_PATH       = "config/anchor.json"
GALLERY_DIR    = "data/gallery"
SKILL_CFG      = "config/skill_cd.json"
OUTPUT_IMG     = "output/result.png"
OUTPUT_JSON    = "output/result.json"
CD_STATE_PATH  = "output/cd_state.json"
COUNTDOWN_DIR_DEFAULT = "data/countdown"

def suppress_by_slots(results, box, cfg):
    """
    对一行的结果按“格”归并：每个格只保留最高分。
    results: [(x, y, label, score), ...]
    返回值: [{"slot": idx, "x": x, "y": y, "label": label, "score": score}, ...]
    box: (start_x, start_y, area_w, area_h) —— 该行检测区域（从第1格或第2格开始，取决于配置）
    """
    start_x, start_y, area_w, area_h = map(int, box)
    step = cfg["cell_w"] + cfg["gap"]
    max_slots = cfg["max_buffs"]

    slots = {}
    for (x, y, lb, sc) in results:
        idx = int(round((x - start_x) / step))
        if idx < 0 or idx >= max_slots:
            continue
        cur = slots.get(idx)
        if (cur is None) or (sc > cur["score"]):
            slots[idx] = {
                "slot": idx,
                "x": int(x),
                "y": int(y),
                "label": lb,
                "score": float(sc)
            }

    # 按槽序输出（稳定顺序）
    return [slots[i] for i in sorted(slots.keys())]

def main(image_path: str):
    # 读取配置与图片
    if not os.path.isfile(CFG_PATH):
        print(f"❌ 找不到配置：{CFG_PATH}")
        return
    cfg = load_json(CFG_PATH)

    if "rows" not in cfg or not cfg["rows"]:
        print("❌ 配置缺少 rows 数组。")
        return

    img = imread_u(image_path)
    if img is None:
        print(f"❌ 无法读取图片：{image_path}")
        return

    # 加载图库（统一到 cell_w/ cell_h）
    gallery = load_gallery(GALLERY_DIR, target_size=(cfg["cell_w"], cfg["cell_h"]))
    if not gallery:
        print(f"❌ 图库为空：{GALLERY_DIR}")
        return

    mask_tl = tuple(cfg.get("mask_topleft", [0, 0]))
    try:
        mask_w = int(mask_tl[0])
        mask_h = int(mask_tl[1])
    except (TypeError, ValueError, IndexError):
        mask_w = mask_h = 0

    countdown_cfg = cfg.get("countdown", {}) or {}
    countdown_enabled = bool(countdown_cfg.get("enabled", True))
    countdown_dir = countdown_cfg.get("gallery", COUNTDOWN_DIR_DEFAULT)
    countdown_min_sim = float(countdown_cfg.get("min_sim", 0.85))
    countdown_gallery = []
    countdown_size = (mask_w, mask_h)

    if countdown_enabled and mask_w > 0 and mask_h > 0:
        countdown_gallery = load_gallery(countdown_dir, target_size=countdown_size)
        if countdown_gallery:
            print(f"ℹ️ 倒计时模板：{len(countdown_gallery)} 个，目录：{countdown_dir}")
        else:
            print(f"⚠️ 倒计时模板为空或未找到：{countdown_dir}")
    elif countdown_enabled:
        print("⚠️ mask_topleft 配置无效，跳过倒计时识别。")

    # 读取/计算固定区域（每行锚一次）
    use_static    = bool(cfg.get("static_regions", True))
    anchor_cachef = cfg.get("anchor_cache", "output/anchor_cache.json")
    areas_by_row, anchors_by_row = {}, {}

    cache = load_anchor_cache(anchor_cachef) if use_static else None
    if cache and cache.get("screen_wh") == [img.shape[1], img.shape[0]]:
        areas_by_row   = cache.get("areas_by_row", {}) or {}
        anchors_by_row = cache.get("anchors_by_row", {}) or {}
        print("ℹ️ 使用缓存的固定区域。")
    else:
        # 调试：锚点可视化
        debug_anchors = bool(cfg.get("debug_anchors", True))
        dbg_a = img.copy() if debug_anchors else None

        for row in cfg["rows"]:
            row_name = row["name"]
            th = float(row.get("anchor_threshold", cfg.get("anchor_threshold", 0.82)))
            m = match_anchor(img, row["anchor_template"], row["search_region"], threshold=th)
            if m is None:
                print(f"⚠️ 行[{row_name}] 未找到锚点（threshold={th}）。跳过该行。")
                continue

            # 兼容老的返回值 (x,y) 或新的 (x,y,w,h)
            if isinstance(m[0], tuple) and len(m[0]) == 2:
                anchor_xy, score = m
                ax, ay = anchor_xy
                aw = cfg["cell_w"]; ah = cfg["cell_h"]
            else:
                (ax, ay, aw, ah), score = m

            anchors_by_row[row_name] = [int(ax), int(ay)]
            if debug_anchors:
                cv2.rectangle(dbg_a, (int(ax), int(ay)), (int(ax+aw), int(ay+ah)), (0, 0, 255), 2)
            print(f"[ANCHOR] 行[{row_name}] score={score:.3f} region={row['search_region']}")

            # 由锚点计算该行的固定检测区域（是否跳过第1格由 cfg 控制）
            box = compute_row_area_from_anchor((ax, ay), row, cfg)  # (x,y,w,h)
            areas_by_row[row_name] = [int(box[0]), int(box[1]), int(box[2]), int(box[3])]

        if debug_anchors and 'dbg_a' in locals() and dbg_a is not None:
            imwrite_u("output/debug_anchors.png", dbg_a)
            print("🔴 已输出锚点调试图：output/debug_anchors.png")

        # 写缓存
        if use_static and areas_by_row:
            save_anchor_cache(anchor_cachef, {
                "screen_wh": [img.shape[1], img.shape[0]],
                "areas_by_row": areas_by_row,
                "anchors_by_row": anchors_by_row
            })
            print(f"💾 已写缓存到 {anchor_cachef}")

    # 行区域调试图（黄框）
    if bool(cfg.get("debug_rows", True)):
        dbg = img.copy()
        for rn, box in areas_by_row.items():
            x, y, w, h = map(int, box)
            cv2.rectangle(dbg, (x, y), (x + w, y + h), (0, 255, 255), 1)
        imwrite_u("output/debug_rows.png", dbg)
        print("🟨 已输出行区域调试图：output/debug_rows.png")

    # 在固定区域内做模板匹配（加入 mask_topleft 支持）
    results_all = {}
    min_sim = float(cfg.get("min_sim_to_draw", 0.74))

    countdown_active = bool(countdown_gallery)

    for row in cfg["rows"]:
        rn = row["name"]
        box = areas_by_row.get(rn)
        if not box:
            continue
        results_raw = recognize_in_box(
            img, box, gallery, min_sim=min_sim, mask_topleft=mask_tl
        )
        results = suppress_by_slots(results_raw, box, cfg)

        if countdown_active:
            slots = enumerate_slots_in_box(box, cfg, countdown_size=countdown_size)
            slot_map = {slot["slot"]: slot for slot in slots}
            countdown_matches = recognize_countdowns_in_slots(
                img, slots, countdown_gallery, min_sim=countdown_min_sim
            )
            for item in results:
                match = countdown_matches.get(item["slot"])
                slot_info = slot_map.get(item["slot"])
                if slot_info:
                    item["countdown_box"] = [
                        int(slot_info["x"]),
                        int(slot_info["y"]),
                        int(slot_info["w"]),
                        int(slot_info["h"])
                    ]
                if match:
                    item["countdown"] = match.get("value")
                    item["countdown_score"] = match.get("score")
                else:
                    item["countdown"] = None
                    item["countdown_score"] = None
        results_all[rn] = results

    # 计算/更新 CD 并输出快照
    snapshot = compute_and_update_cd(results_all, SKILL_CFG, CD_STATE_PATH)
    save_json(OUTPUT_JSON, snapshot)
    print(f"📝 已写入识别/CD 结果：{OUTPUT_JSON}")

    # 可视化输出
    annotate_and_save(img, results_all, OUTPUT_IMG, cfg)
    print(f"🖼️ 已保存可视化结果：{OUTPUT_IMG}")

    # 可选：弹窗显示
    if bool(cfg.get("show_window", False)):
        vis = imread_u(OUTPUT_IMG)
        if vis is not None:
            cv2.imshow("BuffScope Result", vis)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法：python -m app.main <截图路径>")
    else:
        main(sys.argv[1])
