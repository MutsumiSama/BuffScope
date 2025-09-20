# -*- coding: utf-8 -*-
import os, json, cv2, numpy as np

def imread_u(path, flags=cv2.IMREAD_COLOR):
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, flags)

def imwrite_u(path, img):
    d = os.path.dirname(path)
    if d and not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)
    ext = os.path.splitext(path)[1]
    ok, buf = cv2.imencode(ext, img)
    if ok: buf.tofile(path)
    return ok

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path, data):
    d = os.path.dirname(path)
    if d and not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_anchor_cache(path):
    if not os.path.isfile(path): return None
    with open(path, "r", encoding="utf-8") as f:
        try: return json.load(f)
        except: return None

def save_anchor_cache(path, data):
    d = os.path.dirname(path)
    if d and not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
