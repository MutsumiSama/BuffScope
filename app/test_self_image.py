# -*- coding: utf-8 -*-
"""
BuffScope - 最终版
功能：
1. 锚点定位buff栏起点（不参与识别）
2. 从第二个格子开始，做模板匹配识别
3. 每类buff只保留分数最高的一个
4. 标注支持中文，文字上下错开
"""

import os, sys, json
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

CFG_PATH = "config/anchor.json"
GALLERY_DIR = "data/gallery/self"
OUTPUT_PATH = "output/self_result.png"


# ---------- IO ----------
def imread_u(path, flags=cv2.IMREAD_COLOR):
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, flags)

def imwrite_u(path, img):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ext = os.path.splitext(path)[1]
    ok, buf = cv2.imencode(ext, img)
    if ok:
        buf.tofile(path)
    return ok


# ---------- 配置 ----------
def load_cfg():
    with open(CFG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------- 图库 ----------
def load_gallery(gallery_dir, target_size=(32, 32)):
    items = []
    if not os.path.isdir(gallery_dir):
        return items
    for lb in os.listdir(gallery_dir):
        cls_dir = os.path.join(gallery_dir, lb)
        if not os.path.isdir(cls_dir):
            continue
        for name in os.listdir(cls_dir):
            if not name.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            p = os.path.join(cls_dir, name)
            img = imread_u(p)
            if img is None:
                continue
            if target_size:
                img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
            items.append((img, lb, name))
    return items


# ---------- 模板匹配 ----------
def match_one(roi, template):
    if roi.shape[0] < template.shape[0] or roi.shape[1] < template.shape[1]:
        return -1, (0, 0)
    res = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    return max_val, max_loc


def match_anchor(full_img, anchor_path, search_region, threshold=0.82):
    ax, ay, aw, ah = map(int, search_region)
    roi = full_img[ay:ay+ah, ax:ax+aw]
    anchor = imread_u(anchor_path)
    if anchor is None:
        raise FileNotFoundError(f"Anchor template not found: {anchor_path}")
    if roi.shape[0] < anchor.shape[0] or roi.shape[1] < anchor.shape[1]:
        return None, 0.0
    res = cv2.matchTemplate(roi, anchor, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    if max_val < threshold:
        return None, max_val
    x, y = max_loc[0] + ax, max_loc[1] + ay
    return (x, y), max_val


# ---------- 识别逻辑 ----------
def recognize_in_area(full_img, anchor_xy, cfg, gallery):
    x0, y0 = anchor_xy
    w, h = cfg["cell_w"], cfg["cell_h"]
    gap = cfg["gap"]
    max_buffs = cfg["max_buffs"]
    pad_top = int(cfg.get("pad_top", 2))
    pad_bottom = int(cfg.get("pad_bottom", 2))
    min_sim = float(cfg.get("min_sim_to_draw", 0.75))

    # 从第二格开始
    start_x = x0 + (w + gap)
    area_w = w * (max_buffs - 1) + gap * (max_buffs - 2)
    area_h = h + pad_top + pad_bottom
    if area_w <= 0:
        return []

    area = full_img[y0 - pad_top:y0 + h + pad_bottom,
                    start_x:start_x + area_w]

    best_by_class = {}
    for temp, lb, _ in gallery:
        if area.shape[0] < temp.shape[0] or area.shape[1] < temp.shape[1]:
            continue
        res = cv2.matchTemplate(area, temp, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        if max_val < min_sim:
            continue
        abs_x = start_x + max_loc[0]
        abs_y = (y0 - pad_top) + max_loc[1]
        prev = best_by_class.get(lb)
        if (prev is None) or (max_val > prev[0]):
            th, tw = temp.shape[:2]
            best_by_class[lb] = (max_val, abs_x, abs_y, th, tw)

    results = []
    for lb, (score, ax, ay, th, tw) in best_by_class.items():
        results.append((ax, ay, lb, score))
    return results


# ---------- 可视化 ----------
def annotate_and_save(full_img, results, *, anchor_xy=None, anchor_score=None, cfg=None):
    vis = full_img.copy()
    w, h = cfg["cell_w"], cfg["cell_h"]

    # 转成 PIL
    vis_pil = Image.fromarray(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(vis_pil)

    # 字体
    font_path = "C:/Windows/Fonts/msyh.ttc"  # Windows 微软雅黑
    try:
        font = ImageFont.truetype(font_path, 8)
    except:
        font = ImageFont.load_default()

    # 画锚点
    if anchor_xy is not None:
        ax, ay = anchor_xy
        cv2.rectangle(vis, (ax, ay), (ax+w, ay+h), (0, 200, 255), 1)
        if anchor_score is not None:
            draw.text((ax, ay - 18), f"锚点:{anchor_score:.2f}", font=font, fill=(0,200,255,255))

    # 画 buff
    for i, (x, y, lb, score) in enumerate(results):
        cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 1)
        text = f"{lb}:{score:.2f}" if cfg.get("draw_score", False) else lb
        pos = (x, y - 18) if i % 2 == 0 else (x, y + h + 2)
        draw.text(pos, text, font=font, fill=(0,255,0,255))

    vis = cv2.cvtColor(np.array(vis_pil), cv2.COLOR_RGB2BGR)
    imwrite_u(OUTPUT_PATH, vis)
    print(f"✅ 已保存结果到 {OUTPUT_PATH}")


# ---------- 主流程 ----------
def main(image_path):
    if not os.path.isfile(image_path):
        print(f"找不到截图：{image_path}")
        return

    cfg = load_cfg()
    img = imread_u(image_path)
    if img is None:
        print("无法读取图片。")
        return

    anchor_xy, score = match_anchor(img, cfg["anchor_template"], cfg["search_region"], threshold=0.82)
    if anchor_xy is None:
        print("⚠️ 没找到锚点")
        return

    dx, dy = cfg.get("anchor_to_first_cell", [0, 0])
    first_xy = (anchor_xy[0] + dx, anchor_xy[1] + dy)

    gallery = load_gallery(GALLERY_DIR, target_size=(cfg["cell_w"], cfg["cell_h"]))
    if not gallery:
        print("⚠️ 图库为空")
        return

    results = recognize_in_area(img, first_xy, cfg, gallery)
    annotate_and_save(img, results, anchor_xy=anchor_xy, anchor_score=score, cfg=cfg)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法：python app/test_self_image.py <截图路径>")
    else:
        main(sys.argv[1])
