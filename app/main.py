import sys
from .utils import imread_u, load_json, save_json
from .detector import match_anchor, recognize_in_area, load_gallery
from .visualizer import annotate_and_save
from .cd_manager import compute_cd

CFG_PATH = "config/anchor.json"
GALLERY_DIR = "data/gallery"
OUTPUT_IMG = "output/result.png"
OUTPUT_JSON = "output/result.json"
SKILL_CFG = "config/skill_cd.json"

def main(image_path):
    cfg = load_json(CFG_PATH)
    img = imread_u(image_path)
    gallery = load_gallery(GALLERY_DIR, (cfg["cell_w"], cfg["cell_h"]))

    results_all = {}
    for row in cfg["rows"]:
        anchor_xy, score = match_anchor(img, row["anchor_template"], row["search_region"])
        if anchor_xy is None: continue
        dx, dy = row.get("anchor_to_first_cell", [0,0])
        first_xy = (anchor_xy[0]+dx, anchor_xy[1]+dy)
        results = recognize_in_area(img, first_xy, cfg, gallery)
        results_all[row["name"]] = results

    cd_results = compute_cd(results_all, SKILL_CFG)
    save_json(OUTPUT_JSON, cd_results)
    annotate_and_save(img, results_all, OUTPUT_IMG, cfg)

if __name__ == "__main__":
    if len(sys.argv)<2:
        print("用法: python -m app.main <截图路径>")
    else:
        main(sys.argv[1])
