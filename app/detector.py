# -*- coding: utf-8 -*-
import os, cv2
import numpy as np
from .utils import imread_u

# --- 基础匹配 ---
def match_one(roi, template):
    if roi.shape[0] < template.shape[0] or roi.shape[1] < template.shape[1]:
        return -1.0, (0, 0)
    res = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    return float(max_val), max_loc

def match_anchor(full_img, anchor_path, search_region, threshold=0.82):
    ax, ay, aw, ah = map(int, search_region)
    roi = full_img[ay:ay+ah, ax:ax+aw]
    anchor = imread_u(anchor_path)
    if anchor is None: return None, 0.0
    if roi.shape[0] < anchor.shape[0] or roi.shape[1] < anchor.shape[1]:
        return None, 0.0
    res = cv2.matchTemplate(roi, anchor, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    if max_val < threshold: return None, float(max_val)
    x, y = max_loc[0] + ax, max_loc[1] + ay
    return (x, y), float(max_val)

# --- 图库：子目录 or 扁平文件 都支持 ---
def load_gallery(gallery_dir, target_size=(32, 32)):
    items = []
    if not os.path.isdir(gallery_dir): return items
    has_subdirs = any(os.path.isdir(os.path.join(gallery_dir, d)) for d in os.listdir(gallery_dir))
    if has_subdirs:
        for lb in os.listdir(gallery_dir):
            cls_dir = os.path.join(gallery_dir, lb)
            if not os.path.isdir(cls_dir): continue
            for name in os.listdir(cls_dir):
                if not name.lower().endswith((".png", ".jpg", ".jpeg")): continue
                p = os.path.join(cls_dir, name)
                img = imread_u(p)
                if img is None: continue
                if target_size: img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
                items.append((img, lb, name))
    else:
        for name in os.listdir(gallery_dir):
            if not name.lower().endswith((".png", ".jpg", ".jpeg")): continue
            p = os.path.join(gallery_dir, name)
            img = imread_u(p)
            if img is None: continue
            if target_size: img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
            label = os.path.splitext(name)[0]
            items.append((img, label, name))
    return items

# --- 由锚点求“本行固定检测区域”（从第2格开始） ---
def compute_row_area_from_anchor(anchor_xy, row_cfg, global_cfg):
    """
    返回该行用于模板匹配的固定区域 (x, y, w, h)。
    是否跳过第1格由 global_cfg["skip_first_tile"] 控制。
    """
    ax, ay = anchor_xy
    w  = global_cfg["cell_w"];  h  = global_cfg["cell_h"]
    gap = global_cfg["gap"];    maxb = global_cfg["max_buffs"]
    pad_t = int(global_cfg.get("pad_top", 2))
    pad_b = int(global_cfg.get("pad_bottom", 2))
    skip_first = bool(global_cfg.get("skip_first_tile", False))

    dx0, dy0 = row_cfg.get("anchor_to_first_cell",
                           global_cfg.get("anchor_to_first_cell", [0, 0]))
    dx0, dy0 = int(dx0), int(dy0)

    # 第一个buff格子的左上角
    first_x = ax + dx0
    first_y = ay + dy0

    # 起点：包含首格 or 跳过首格
    if skip_first:
        start_x = first_x + (w + gap)
        area_w  = max(0, w * (maxb - 1) + gap * (maxb - 2))
    else:
        start_x = first_x
        area_w  = w * maxb + gap * (maxb - 1)

    start_y = first_y - pad_t
    area_h  = h + pad_t + pad_b
    return (start_x, start_y, area_w, area_h)

# --- 在已知区域内做模板匹配（每类保留最高分） ---
def recognize_in_box(full_img, box_xywh, gallery, min_sim=0.75, mask_topleft=(0,0)):
    """
    已知固定区域 box=(x,y,w,h)，对图库逐一模板匹配；每类只留最高分。
    mask_topleft: (mw, mh) 屏蔽模板左上角宽mw*高mh的区域（覆盖倒计时数字）
    """
    x, y, w, h = map(int, box_xywh)
    if w <= 0 or h <= 0: return []
    area = full_img[y:y+h, x:x+w]

    mw, mh = (int(mask_topleft[0]), int(mask_topleft[1])) if mask_topleft else (0, 0)
    use_mask = (mw > 0 and mh > 0)

    best_by_class = {}
    for temp, lb, _ in gallery:
        if area.shape[0] < temp.shape[0] or area.shape[1] < temp.shape[1]:
            continue

        if use_mask:
            # 构造同尺寸的模板掩模：不屏蔽=255，屏蔽=0
            mask = np.ones(temp.shape[:2], dtype=np.uint8) * 255
            mask[0:mh, 0:mw] = 0
            # 掩模仅支持部分方法：用 TM_CCORR_NORMED
            res = cv2.matchTemplate(area, temp, cv2.TM_CCORR_NORMED, mask=mask)
        else:
            res = cv2.matchTemplate(area, temp, cv2.TM_CCOEFF_NORMED)

        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        if max_val < float(min_sim):
            continue

        abs_x = x + max_loc[0]
        abs_y = y + max_loc[1]
        prev = best_by_class.get(lb)
        if (prev is None) or (max_val > prev[0]):
            best_by_class[lb] = (float(max_val), abs_x, abs_y)

    results = []
    for lb, (score, ax, ay) in best_by_class.items():
        results.append((ax, ay, lb, score))
    return results


def enumerate_slots_in_box(box_xywh, global_cfg, countdown_size=None):
    """根据固定检测区域，推算每个格子的标准位置。"""
    start_x, start_y, area_w, _ = map(int, box_xywh)
    cell_w = int(global_cfg.get("cell_w", 0))
    cell_h = int(global_cfg.get("cell_h", 0))
    gap = int(global_cfg.get("gap", 0))
    max_slots = int(global_cfg.get("max_buffs", 0))
    pad_top = int(global_cfg.get("pad_top", 0))

    if countdown_size is None:
        mask = global_cfg.get("mask_topleft", [0, 0])
        try:
            mask_w, mask_h = int(mask[0]), int(mask[1])
        except (TypeError, ValueError, IndexError):
            mask_w = mask_h = 0
    else:
        mask_w, mask_h = map(int, countdown_size)

    if mask_w <= 0 or mask_h <= 0:
        mask_w, mask_h = cell_w, cell_h

    slots = []
    if cell_w <= 0 or cell_h <= 0 or max_slots <= 0:
        return slots

    step = cell_w + gap
    area_right = start_x + area_w
    for idx in range(max_slots):
        slot_x = start_x + idx * step
        if slot_x + cell_w > area_right:
            break
        slots.append({
            "slot": idx,
            "x": slot_x,
            "y": start_y + pad_top,
            "w": mask_w,
            "h": mask_h
        })
    return slots


def recognize_countdowns_in_slots(full_img, slots, countdown_gallery, min_sim=0.8):
    """在每个格子的倒计时区域内做模板匹配，返回识别到的数字。"""
    results = {}
    if not countdown_gallery or not slots:
        return results

    for slot in slots:
        x, y, w, h = slot["x"], slot["y"], slot["w"], slot["h"]
        roi = full_img[y:y+h, x:x+w]
        if roi.shape[0] < h or roi.shape[1] < w:
            continue

        best = None
        for temp, lb, _ in countdown_gallery:
            if roi.shape[0] < temp.shape[0] or roi.shape[1] < temp.shape[1]:
                continue
            res = cv2.matchTemplate(roi, temp, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(res)
            if best is None or max_val > best[1]:
                best = (lb, float(max_val))

        if not best:
            continue

        label, score = best
        if score < float(min_sim):
            continue

        try:
            value = int(label)
        except (TypeError, ValueError):
            try:
                value = int(str(label).strip())
            except (TypeError, ValueError):
                value = label

        results[slot["slot"]] = {
            "label": label,
            "value": value,
            "score": score
        }

    return results
