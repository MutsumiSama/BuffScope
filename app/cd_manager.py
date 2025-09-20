# -*- coding: utf-8 -*-
import os, json, time, datetime
from typing import Dict, List, Tuple, Optional

def _now_ts(): return time.time()
def _fmt(ts):  return datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")

def _ensure_dir(path):
    d = os.path.dirname(path)
    if d and not os.path.isdir(d): os.makedirs(d, exist_ok=True)

def load_skill_map(path: str) -> Dict[str, Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_cd_state(path: str) -> Dict:
    if not os.path.isfile(path): return {"rows": {}}
    with open(path, "r", encoding="utf-8") as f:
        try: return json.load(f)
        except: return {"rows": {}}

def save_cd_state(path: str, state: Dict) -> None:
    _ensure_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

def compute_and_update_cd(results_all: Dict[str, List[Tuple[int,int,str,float]]],
                          skill_map_path: str,
                          state_path: str,
                          now_ts: Optional[float] = None) -> Dict:
    skill_map = load_skill_map(skill_map_path)
    state = load_cd_state(state_path); state.setdefault("rows", {})
    now = float(now_ts) if (now_ts is not None) else _now_ts()

    snapshot = {"time": _fmt(now), "rows": {}}
    for row_name, detects in results_all.items():
        row_state = state["rows"].setdefault(row_name, {})  # {skill:{start,end,buff}}
        # 触发/刷新
        for _, _, label, _ in detects:
            if label not in skill_map:
                continue
            info = skill_map[label]
            skill = str(info.get("skill", label))
            cd = float(info.get("cd", 0))
            if cd <= 0:
                continue
            rec = row_state.get(skill)
            if (rec is None) or (now >= rec.get("end", 0.0)):
                row_state[skill] = {
                    "start": now,
                    "end": now + cd,
                    "buff": label
                }

        # 当前快照
        active = []
        for skill, rec in row_state.items():
            if now < rec["end"]:
                active.append({
                    "buff": rec.get("buff", ""),
                    "skill": skill,
                    "cd": int(round(rec["end"] - rec["start"])),
                    "start_at": _fmt(rec["start"]),
                    "expire_at": _fmt(rec["end"]),
                    "remain": int(max(0, round(rec["end"] - now)))
                })
        active.sort(key=lambda x: x["expire_at"])
        snapshot["rows"][row_name] = active

    save_cd_state(state_path, state)
    return snapshot
