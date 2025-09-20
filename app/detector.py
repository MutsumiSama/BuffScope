import cv2
from .utils import imread_u, load_json
import os

# app/detector.py 片段
import os
import cv2
from .utils import imread_u

def load_gallery(gallery_dir, target_size=(32, 32)):
    """
    统一加载图库：
    - 若目录下存在子文件夹：每个子文件夹名是类别，内部任意图片皆为该类别的模板
    - 若目录下是扁平文件：每个文件名(去扩展名)就是类别，仅用该文件作为模板
    返回: [(img, label, filename), ...]  其中 label 用于 skill_cd.json 的键
    """
    items = []
    if not os.path.isdir(gallery_dir):
        return items

    has_subdirs = any(os.path.isdir(os.path.join(gallery_dir, d)) for d in os.listdir(gallery_dir))

    if has_subdirs:
        # 传统：多文件夹（每类可多图）
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
    else:
        # 扁平：每个文件就是一个类别（文件名=类别）
        for name in os.listdir(gallery_dir):
            if not name.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            p = os.path.join(gallery_dir, name)
            img = imread_u(p)
            if img is None:
                continue
            if target_size:
                img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
            label = os.path.splitext(name)[0]  # 文件名 = 类别
            items.append((img, label, name))

    return items



def match_one(roi, template):
    res = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    return max_val, max_loc

def match_anchor(full_img, anchor_path, search_region, threshold=0.82):
    ax, ay, aw, ah = map(int, search_region)
    roi = full_img[ay:ay+ah, ax:ax+aw]
    anchor = imread_u(anchor_path)
    if anchor is None:
        return None, 0.0
    res = cv2.matchTemplate(roi, anchor, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    if max_val < threshold:
        return None, max_val
    x, y = max_loc[0] + ax, max_loc[1] + ay
    return (x, y), max_val

def recognize_in_area(full_img, first_xy, cfg, gallery):
    x0, y0 = first_xy
    w, h = cfg["cell_w"], cfg["cell_h"]
    gap = cfg["gap"]
    max_buffs = cfg["max_buffs"]
    pad_top = cfg.get("pad_top", 2)
    pad_bottom = cfg.get("pad_bottom", 2)
    min_sim = cfg.get("min_sim_to_draw", 0.75)

    start_x = x0 + (w + gap)   # 跳过锚点格子
    area_w = w * (max_buffs - 1) + gap * (max_buffs - 2)
    area_h = h + pad_top + pad_bottom
    area = full_img[y0 - pad_top:y0 + h + pad_bottom,
                    start_x:start_x + area_w]

    best_by_class = {}
    for temp, lb, _ in gallery:
        if area.shape[0] < temp.shape[0] or area.shape[1] < temp.shape[1]:
            continue
        score, loc = match_one(area, temp)
        if score < min_sim: continue
        abs_x, abs_y = start_x + loc[0], (y0 - pad_top) + loc[1]
        if lb not in best_by_class or score > best_by_class[lb][0]:
            best_by_class[lb] = (score, abs_x, abs_y, temp.shape[:2])

    results = []
    for lb, (score, ax, ay, (th, tw)) in best_by_class.items():
        results.append((ax, ay, lb, score))
    return results
