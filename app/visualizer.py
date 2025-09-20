# -*- coding: utf-8 -*-
import cv2, numpy as np
from PIL import Image, ImageDraw, ImageFont
from .utils import imwrite_u

def annotate_and_save(full_img, results_all, output_path, cfg):
    vis = full_img.copy()
    w, h = cfg["cell_w"], cfg["cell_h"]

    # 转 PIL 绘字（兼容中文/英文）
    vis_pil = Image.fromarray(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(vis_pil)
    font_path = cfg.get("font_path", "C:/Windows/Fonts/msyh.ttc")
    try:
        font = ImageFont.truetype(font_path, 16)
    except:
        font = ImageFont.load_default()

    for row_name, results in results_all.items():
        for i, (x, y, lb, score) in enumerate(results):
            # 画框
            cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 1)
            # 文件名直接显示
            text = f"{row_name}:{lb}:{score:.2f}" if cfg.get("draw_score", False) \
                   else f"{row_name}:{lb}"
            pos = (x, y - 18) if i % 2 == 0 else (x, y + h + 2)
            draw.text(pos, text, font=font, fill=(0,255,0,255))

    vis = cv2.cvtColor(np.array(vis_pil), cv2.COLOR_RGB2BGR)
    imwrite_u(output_path, vis)
