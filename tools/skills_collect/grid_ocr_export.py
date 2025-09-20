import os
import cv2
import argparse
import shutil
import numpy as np
from PIL import Image

# ========= 固定参数 =========
ICON_W = 67
ICON_H = 67
STEP_X = 221      # 横向步长
STEP_Y = 75       # 纵向步长
COLS   = 4        # 每行图标数

def is_duplicate(img1, img2, tol=0):
    """判断两张图是否重复，tol=允许不同的像素数量"""
    if img1.shape != img2.shape:
        return False
    diff = cv2.absdiff(img1, img2)
    nonzero = np.count_nonzero(diff)
    return nonzero <= tol

def extract_from_image(img_path, out_dir, start_idx):
    """从单张截图里提取图标"""
    img = cv2.imread(img_path)
    if img is None:
        print(f"无法读取: {img_path}")
        return start_idx

    H, W = img.shape[:2]

    # 自动推断行数
    rows = 0
    while True:
        iy = rows * STEP_Y
        if iy + ICON_H > H:
            break
        rows += 1

    count = 0
    for r in range(rows):
        for c in range(COLS):
            ix = c * STEP_X
            iy = r * STEP_Y
            if ix + ICON_W > W or iy + ICON_H > H:
                continue

            icon = img[iy:iy+ICON_H, ix:ix+ICON_W].copy()

            idx = start_idx + count + 1
            save_name = f"skill_{idx:04d}.png"
            save_path = os.path.join(out_dir, save_name)
            Image.fromarray(cv2.cvtColor(icon, cv2.COLOR_BGR2RGB)).save(save_path)

            count += 1
    print(f"{img_path}: 提取 {count} 个图标")
    return start_idx + count

def deduplicate_images(out_dir, tol=0, move=False):
    """像素级去重"""
    files = [f for f in os.listdir(out_dir) if f.lower().endswith(".png")]
    files.sort()
    kept = []
    removed = 0
    dup_dir = os.path.join(out_dir, "duplicates")
    if move:
        os.makedirs(dup_dir, exist_ok=True)

    for f in files:
        fpath = os.path.join(out_dir, f)
        img = cv2.imread(fpath)
        if img is None:
            continue

        duplicate_found = False
        for (keep_name, keep_img) in kept:
            if is_duplicate(img, keep_img, tol):
                duplicate_found = True
                if move:
                    shutil.move(fpath, os.path.join(dup_dir, f))
                    print(f"移动重复: {f}")
                else:
                    os.remove(fpath)
                    print(f"删除重复: {f}")
                removed += 1
                break

        if not duplicate_found:
            kept.append((f, img))

    print(f"\n去重完成！删除/移动 {removed} 个重复，保留 {len(kept)} 个唯一图标。")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="输入截图文件夹")
    ap.add_argument("--out", default="skills_out", help="输出目录")
    ap.add_argument("--tol", type=int, default=0, help="像素差异容忍度 (默认=0 完全一致才算重复)")
    ap.add_argument("--move", action="store_true", help="将重复移动到 duplicates 文件夹，而不是直接删除")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    idx = 0
    for fname in os.listdir(args.input):
        if fname.lower().endswith((".png", ".jpg", ".jpeg")):
            img_path = os.path.join(args.input, fname)
            idx = extract_from_image(img_path, args.out, idx)

    deduplicate_images(args.out, tol=args.tol, move=args.move)
    print(f"\n全部完成！共提取 {idx} 个图标（去重前）。")

if __name__ == "__main__":
    main()
