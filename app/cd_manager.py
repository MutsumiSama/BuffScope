# -*- coding: utf-8 -*-
import os, json, time, datetime
from typing import Dict, List, Tuple, Optional, Any

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

def _to_float(val: Any) -> Optional[float]:
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _sec(val: Optional[float]) -> int:
    if val is None:
        return 0
    return int(max(0, round(val)))


def _resolve_timer(timer: Optional[Dict[str, Any]],
                   fallback_start: Optional[float],
                   fallback_end: Optional[float],
                   now: float) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    if timer is None:
        start = fallback_start
        end = fallback_end
        value = None
    else:
        start = _to_float(timer.get("start", fallback_start))
        end = _to_float(timer.get("end", fallback_end))
        value = _to_float(timer.get("value"))

    if value is None and start is not None and end is not None:
        value = end - start

    remain = None
    if end is not None:
        remain = max(0.0, end - now)

    return value, start, end, remain


def _is_timer_active(timer: Optional[Dict[str, Any]],
                     fallback_end: Optional[float],
                     now: float) -> bool:
    if timer is None:
        end = fallback_end
    else:
        end = _to_float(timer.get("end", fallback_end))
    return bool(end is not None and now < end)


def _start_skill_record(row_state: Dict[str, Dict[str, Any]],
                        label: str,
                        info: Dict[str, Any],
                        now: float,
                        countdown: Optional[Any] = None,
                        force: bool = False) -> None:
    skill_name = str(info.get("skill", label))
    cd_val = _to_float(info.get("cd")) or 0.0
    duration_val = _to_float(info.get("duration")) or 0.0

    if isinstance(countdown, dict):
        countdown_val = _to_float(countdown.get("value"))
        if countdown_val is None:
            countdown_val = _to_float(countdown.get("remain"))
    else:
        countdown_val = _to_float(countdown)

    if cd_val <= 0 and duration_val <= 0:
        return

    rec = row_state.get(skill_name, {})

    cd_active = False
    duration_active = False
    has_countdown = countdown_val is not None
    if rec and not force and not has_countdown:
        cd_active = _is_timer_active(rec.get("cd"), _to_float(rec.get("end")), now) if cd_val > 0 else False
        duration_active = _is_timer_active(rec.get("duration"), None, now) if duration_val > 0 else False
        if cd_active or duration_active:
            rec["buff"] = label  # 刷新 buff 名称
            row_state[skill_name] = rec
            return

    rec["buff"] = label
    rec["skill"] = skill_name

    remain = None
    if countdown_val is not None:
        remain = max(0.0, countdown_val)
        if duration_val > 0:
            remain = min(duration_val, remain)

    base_start = now

    if duration_val > 0:
        if remain is not None:
            duration_start = now + remain - duration_val
            duration_end = duration_start + duration_val
        else:
            duration_start = now
            duration_end = now + duration_val
        rec["duration"] = {
            "start": duration_start,
            "end": duration_end,
            "value": duration_val
        }
        base_start = duration_start
    else:
        rec.pop("duration", None)

    if remain is not None:
        rec["last_countdown"] = remain
    elif countdown_val is not None:
        rec["last_countdown"] = max(0.0, countdown_val)

    if cd_val > 0:
        cd_start = base_start
        cd_end = cd_start + cd_val
        rec["cd"] = {"start": cd_start, "end": cd_end, "value": cd_val}
        rec["start"] = cd_start
        rec["end"] = cd_end
    else:
        rec.pop("cd", None)
        rec.pop("start", None)
        rec.pop("end", None)

    row_state[skill_name] = rec


def _build_snapshot_entry(skill: str,
                          rec: Dict[str, Any],
                          now: float) -> Optional[Tuple[float, Dict[str, Any]]]:
    cd_val, cd_start, cd_end, cd_remain = _resolve_timer(
        rec.get("cd"),
        _to_float(rec.get("start")),
        _to_float(rec.get("end")),
        now
    )
    duration_info = rec.get("duration")
    duration_val, duration_start, duration_end, duration_remain = _resolve_timer(
        duration_info,
        _to_float(duration_info.get("start")) if isinstance(duration_info, dict) else None,
        _to_float(duration_info.get("end")) if isinstance(duration_info, dict) else None,
        now
    )

    active = False
    if cd_remain is not None and cd_remain > 0:
        active = True
    if duration_remain is not None and duration_remain > 0:
        active = True

    if not active:
        return None

    start_at = cd_start if cd_start is not None else duration_start
    entry = {
        "buff": rec.get("buff", ""),
        "skill": rec.get("skill", skill),
        "cd": _sec(cd_val),
        "start_at": _fmt(start_at) if start_at is not None else None,
        "expire_at": _fmt(cd_end) if cd_end is not None else None,
        "remain": _sec(cd_remain)
    }

    entry["duration"] = _sec(duration_val)
    entry["duration_start_at"] = _fmt(duration_start) if duration_start is not None else None
    entry["duration_expire_at"] = _fmt(duration_end) if duration_end is not None else None
    entry["duration_remain"] = _sec(duration_remain)

    last_countdown = rec.get("last_countdown")
    if last_countdown is not None:
        entry["last_countdown"] = _sec(last_countdown)

    sort_candidates = [ts for ts in (cd_end, duration_end) if ts is not None]
    sort_ts = min(sort_candidates) if sort_candidates else float("inf")
    return sort_ts, entry


def _build_rows_snapshot(rows_state: Dict[str, Dict[str, Dict[str, Any]]],
                         now: float) -> Dict[str, List[Dict[str, Any]]]:
    rows_snapshot: Dict[str, List[Dict[str, Any]]] = {}
    for row_name, row_state in rows_state.items():
        entries: List[Tuple[float, Dict[str, Any]]] = []
        for skill, rec in list(row_state.items()):
            snapshot_item = _build_snapshot_entry(skill, rec, now)
            if snapshot_item is None:
                continue
            entries.append(snapshot_item)
        entries.sort(key=lambda item: item[0])
        rows_snapshot[row_name] = [entry for _, entry in entries]
    return rows_snapshot


def compute_and_update_cd(results_all: Dict[str, List[Any]],
                          skill_map_path: str,
                          state_path: str,
                          now_ts: Optional[float] = None) -> Dict:
    skill_map = load_skill_map(skill_map_path)
    state = load_cd_state(state_path)
    state.setdefault("rows", {})
    now = float(now_ts) if (now_ts is not None) else _now_ts()

    for row_name, detects in results_all.items():
        row_state = state["rows"].setdefault(row_name, {})
        for detect in detects:
            if isinstance(detect, dict):
                label = detect.get("label")
                countdown_val = detect.get("countdown")
            else:
                if len(detect) < 3:
                    continue
                label = detect[2]
                countdown_val = None
            if not label:
                continue
            info = skill_map.get(label)
            if not info:
                continue
            _start_skill_record(row_state, label, info, now, countdown=countdown_val)

    snapshot_rows = _build_rows_snapshot(state["rows"], now)
    snapshot = {"time": _fmt(now), "rows": snapshot_rows}

    save_cd_state(state_path, state)
    return snapshot


def start_skill_cd(row_name: str,
                   buff_label: str,
                   skill_map_path: str,
                   state_path: str,
                   now_ts: Optional[float] = None,
                   force: bool = False) -> Dict:
    """手动触发某一格的技能冷却/持续时间计算。

    Args:
        row_name: 行名称（例如“云”“剑”“斧”）。
        buff_label: 识别到的 buff 名称，与 skill_cd.json 中的键一致。
        skill_map_path: 技能配置文件路径。
        state_path: 冷却状态文件路径。
        now_ts: 可选的当前时间戳（秒）。
        force: 若为 True，无视当前记录强制重新开始计时。

    Returns:
        与 compute_and_update_cd 相同结构的快照字典。
    """
    skill_map = load_skill_map(skill_map_path)
    info = skill_map.get(buff_label)
    if not info:
        raise KeyError(f"未在配置中找到技能：{buff_label}")

    state = load_cd_state(state_path)
    state.setdefault("rows", {})
    row_state = state["rows"].setdefault(row_name, {})
    now = float(now_ts) if (now_ts is not None) else _now_ts()

    _start_skill_record(row_state, buff_label, info, now, force=force)

    snapshot_rows = _build_rows_snapshot(state["rows"], now)
    snapshot = {"time": _fmt(now), "rows": snapshot_rows}

    save_cd_state(state_path, state)
    return snapshot
