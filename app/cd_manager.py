import datetime
from .utils import load_json

def compute_cd(results_all, skill_cfg_path):
    skill_map = load_json(skill_cfg_path)
    now = datetime.datetime.now()
    cd_results = {}

    for row_name, results in results_all.items():
        cd_results[row_name] = []
        for _, _, lb, _ in results:
            if lb in skill_map:
                skill = skill_map[lb]["skill"]
                cd = skill_map[lb]["cd"]
                expire_at = (now + datetime.timedelta(seconds=cd)).strftime("%H:%M:%S")
                cd_results[row_name].append({
                    "buff": lb, "skill": skill, "cd": cd, "expire_at": expire_at
                })
    return cd_results
