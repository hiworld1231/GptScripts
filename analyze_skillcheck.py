#!/usr/bin/env python3
import argparse
import json
import math
import os
import re
import subprocess
import tempfile
import fcntl
from pathlib import Path

import cv2
import numpy as np

REPORT_LIMIT_LINES = 500
REPORT_ALL = "analysis_all.json"
MERGE_LOCK = ".analysis_merge.lock"


def newest():
    files = [x for x in os.listdir(".") if re.fullmatch(r"session_\d+\.mkv", x)]
    if not files:
        raise FileNotFoundError("Не найден session_*.mkv")
    return max(files, key=lambda x: int(re.search(r"\d+", x).group()))


def parse_ass_time(value):
    m = re.match(r"^(\d+):(\d+):(\d+)[.:](\d+)$", value.strip())
    if not m:
        return None
    h, mi, s, fraction = m.groups()
    fraction = fraction[:3].ljust(3, "0")
    return int(h) * 3600 + int(mi) * 60 + int(s) + int(fraction) / 1000.0


def events(path):
    q = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", path, "-map", "0:s:0", "-f", "ass", "-"], capture_output=True, text=True)
    out = []
    for line in q.stdout.splitlines():
        if not line.startswith("Dialogue:"):
            continue
        z = line.split(",", 9)
        if len(z) < 10:
            continue
        t = parse_ass_time(z[1])
        if t is None:
            continue
        k = re.search(r"(LMB_DOWN|LMB_UP|SPACE_DOWN|SPACE_UP)", z[9])
        if k:
            out.append((t, k.group(1)))
    return sorted(out)


def probe(path):
    q = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,r_frame_rate,duration", "-of", "json", path], capture_output=True, text=True)
    return json.loads(q.stdout)["streams"][0]


def hough(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    variants = [
        cv2.GaussianBlur(gray, (5, 5), 1.0),
        cv2.GaussianBlur(gray, (7, 7), 1.5),
        cv2.GaussianBlur(gray, (9, 9), 2.0),
        cv2.medianBlur(gray, 5),
    ]
    params = (
        (1.00, 80, 20), (1.05, 85, 21), (1.10, 90, 22),
        (1.15, 90, 23), (1.20, 100, 25), (1.25, 105, 27),
        (1.30, 110, 28), (1.35, 115, 30), (1.40, 120, 31),
    )
    out = []
    for g in variants:
        for dp, p1, p2 in params:
            cs = cv2.HoughCircles(g, cv2.HOUGH_GRADIENT, dp=dp, minDist=30, param1=p1, param2=p2, minRadius=40, maxRadius=100)
            if cs is not None:
                out.extend((float(x), float(y), float(r)) for x, y, r in cs[0])
    unique = []
    for c in out:
        if not any(math.hypot(c[0] - q[0], c[1] - q[1]) < 7 and abs(c[2] - q[2]) < 7 for q in unique):
            unique.append(c)
    return unique


def ring_features(frame, cx, cy, r, bins=180):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0].astype(np.float32)
    s = hsv[:, :, 1].astype(np.float32)
    v = hsv[:, :, 2].astype(np.float32)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    yy, xx = np.indices(v.shape)
    rr = np.hypot(xx - cx, yy - cy)
    keep = (rr >= r - 8) & (rr <= r + 11)
    if int(keep.sum()) < 150:
        return None
    ang = (np.arctan2(-(yy - cy), xx - cx) + 2 * np.pi) % (2 * np.pi)
    bi = (ang[keep] * bins / (2 * np.pi)).astype(np.int32)
    n = np.bincount(bi, minlength=bins)

    def hist(values):
        return np.bincount(bi, weights=values[keep], minlength=bins) / np.maximum(n, 1)

    vv = v[keep]
    ss = s[keep]
    v20, v35, _, v65, v80 = np.percentile(vv, [20, 35, 50, 65, 80])
    s20, _, s80 = np.percentile(ss, [20, 50, 80])
    white = np.clip((v - (v65 + 3)) / max(1, 255 - v65), 0, 1) * np.clip((s80 + 25 - s) / max(20, s80 + 25), 0, 1)
    black = np.clip((v35 - v) / max(20, v35 - v20 + 20), 0, 1) * (0.35 + 0.65 * np.clip((s + 15) / 100, 0, 1))
    red = (np.minimum(np.abs(h), np.abs(h - 180)) < 16).astype(np.float32) * np.clip((s - s20) / max(20, 100 - s20), 0, 1)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge = np.hypot(gx, gy)
    return hist(white), hist(black), hist(red), hist(np.clip(edge / 80, 0, 1))


def smooth(x, n=5):
    k = np.ones(2 * n + 1, np.float32) / (2 * n + 1)
    return np.convolve(np.r_[x[-n:], x, x[:n]], k, "same")[n:-n]


def peak(a):
    a = smooth(a)
    i = int(np.argmax(a))
    mean = float(a.mean())
    sd = float(a.std()) + 1e-6
    return (i + 0.5) * 2 * np.pi / len(a), max(0.0, (float(a[i]) - mean) / sd), float(a[i])


def circular_distance(a, b):
    return math.atan2(math.sin(a - b), math.cos(a - b))


def candidate_score(frame, candidate, previous=None):
    f = ring_features(frame, *candidate)
    if f is None:
        return -1.0, None
    wh, bh, rh, eh = f
    wa, ws, wp = peak(wh)
    ba, bs, bp = peak(bh)
    ra, rs, rp = peak(rh)
    ea, es, ep = peak(eh)
    x, y, r = candidate
    score = max(ws, bs) + 0.45 * rs + 0.25 * es + 0.15 * max(wp, bp)
    if previous is not None:
        px, py, pr = previous
        score -= 0.035 * math.hypot(x - px, y - py) + 0.05 * abs(r - pr)
    return score, (wa, ws, wp, ba, bs, bp, ra, rs, rp, ea, es, ep)


def detect_frame(frame, previous=None):
    ranked = []
    for candidate in hough(frame):
        score, features = candidate_score(frame, candidate, previous)
        if score >= 0:
            ranked.append((score, candidate, features))
    if not ranked:
        return None
    ranked.sort(key=lambda x: x[0], reverse=True)
    return ranked[0]


def local_check(cap, fps, start, end, sample_fps):
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0, start * 1000))
    first = max(0, int(round(start * fps)))
    last = int(round(end * fps))
    step = max(1, round(fps / sample_fps))
    frame_i = first
    previous = None
    rows = []
    while frame_i <= last:
        ok, frame = cap.read()
        if not ok:
            break
        if (frame_i - first) % step:
            frame_i += 1
            continue
        hit = detect_frame(frame, previous)
        if hit is not None:
            score, c, feat = hit
            if previous is None or math.hypot(c[0] - previous[0], c[1] - previous[1]) < 70:
                previous = c
                rows.append({"t": frame_i / fps, "x": c[0], "y": c[1], "r": c[2], "score": score, "white_angle": feat[0], "white_strength": feat[1], "white_peak": feat[2], "black_angle": feat[3], "black_strength": feat[4], "black_peak": feat[5], "red_angle": feat[6], "red_strength": feat[7], "red_peak": feat[8], "edge_angle": feat[9], "edge_strength": feat[10], "edge_peak": feat[11]})
        frame_i += 1
    return rows


def fit_red(rows, space):
    good = [x for x in rows if x["t"] <= space + 0.03 and x["red_strength"] > 0.25]
    if len(good) < 8:
        return None
    good = good[-45:]
    t = np.array([x["t"] for x in good], dtype=np.float64)
    a = np.unwrap(np.array([x["red_angle"] for x in good], dtype=np.float64))
    z = t - space
    best = None
    for degree in (1, 2):
        if len(t) < degree + 3:
            continue
        try:
            coef = np.polyfit(z, a, degree)
            pred = float(np.polyval(coef, 0.0))
            velocity = float(np.polyval(np.polyder(coef), 0.0))
            residual = float(np.sqrt(np.mean((np.polyval(coef, z) - a) ** 2)))
            candidate = (residual, pred, velocity, degree)
            if best is None or candidate[0] < best[0]:
                best = candidate
        except Exception:
            pass
    if best is None:
        return None
    residual, predicted, velocity, degree = best
    return {"predicted_angle": predicted, "speed_deg_s": math.degrees(velocity), "fit_residual_deg": math.degrees(residual), "degree": degree, "points": len(good)}


def angular_mean(values):
    if not values:
        return None
    return float(np.angle(np.mean(np.exp(1j * np.array(values)))) % (2 * np.pi))


def summarize(rows, space):
    if not rows:
        return {"result": "NO_CIRCLE", "frames": 0}
    before = [x for x in rows if x["t"] <= space + 0.03]
    recent = before[-30:] if before else rows[-30:]
    red = fit_red(rows, space)
    white_angle = angular_mean([x["white_angle"] for x in recent])
    black_angle = angular_mean([x["black_angle"] for x in recent])
    white_strength = float(np.median([x["white_strength"] for x in recent]))
    black_strength = float(np.median([x["black_strength"] for x in recent]))
    if red is not None and white_angle is not None and black_angle is not None:
        wd = abs(math.degrees(circular_distance(red["predicted_angle"], white_angle)))
        bd = abs(math.degrees(circular_distance(red["predicted_angle"], black_angle)))
        margin = abs(wd - bd)
        if margin >= 8.0:
            result = "WHITE" if wd < bd else "BLACK"
        else:
            result = "UNCERTAIN"
        target_distance = min(wd, bd)
    else:
        ratio = white_strength / max(black_strength, 1e-6)
        if ratio >= 1.35:
            result = "WHITE"
        elif ratio <= 0.74:
            result = "BLACK"
        else:
            result = "UNCERTAIN"
        target_distance = None
        wd = bd = None
    return {"result": result, "frames": len(rows), "first_t": rows[0]["t"], "last_t": rows[-1]["t"], "center_mean": [float(np.mean([x["x"] for x in rows])), float(np.mean([x["y"] for x in rows]))], "radius_mean": float(np.mean([x["r"] for x in rows])), "white_angle_deg": math.degrees(white_angle) if white_angle is not None else None, "black_angle_deg": math.degrees(black_angle) if black_angle is not None else None, "white_strength": white_strength, "black_strength": black_strength, "red_prediction": red, "white_distance_deg": wd, "black_distance_deg": bd, "target_distance_deg": target_distance}


def write_atomic_json(path, data):
    fd, temp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent), text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def line_count(path):
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def merge_reports(session_report):
    lock_fd = os.open(MERGE_LOCK, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        all_path = Path(REPORT_ALL)
        merged = {"version": 2, "sessions": []}
        if all_path.exists():
            try:
                old = json.loads(all_path.read_text(encoding="utf-8"))
                if isinstance(old, dict) and isinstance(old.get("sessions"), list):
                    merged = old
            except Exception:
                pass
        merged["sessions"].append(session_report)
        legacy = sorted(Path(".").glob("analysis_report*.json"))
        for path in legacy:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                merged.setdefault("legacy_chunks", []).append({"file": path.name, "payload": payload})
                path.unlink(missing_ok=True)
            except Exception:
                pass
        write_atomic_json(all_path, merged)
        if session_report.get("file"):
            per_session = Path(session_report["file"])
            per_session.unlink(missing_ok=True)
        print(f"JSON: объединено в {REPORT_ALL}, строк={line_count(all_path)}")
        if line_count(all_path) > REPORT_LIMIT_LINES:
            print(f"JSON: предупреждение — {REPORT_ALL} больше {REPORT_LIMIT_LINES} строк")
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file", nargs="?")
    ap.add_argument("--sample-fps", type=float, default=60.0)
    ap.add_argument("--before", type=float, default=2.5)
    ap.add_argument("--after", type=float, default=0.2)
    args = ap.parse_args()
    path = args.file or newest()
    print(f"===== АНАЛИЗ {path} =====")
    print(f"Файл: {path}")
    meta = probe(path)
    print(f"Видео: {meta['width']}x{meta['height']} fps={meta['r_frame_rate']}")
    ev = events(path)
    spaces = [t for t, k in ev if k == "SPACE_DOWN"]
    print(f"SPACE: {len(spaces)}")
    print("SPACE times: " + ", ".join(f"{t:.3f}" for t in spaces))
    print(f"Анализ: {args.sample_fps:g} FPS, каждый кадр, 36 Hough-проходов, adaptive HSV, random center")
    cap = cv2.VideoCapture(path)
    checks = []
    for i, space in enumerate(spaces, 1):
        rows = local_check(cap, float(eval_fraction(meta["r_frame_rate"])), max(0, space - args.before), space + args.after, args.sample_fps)
        summary = summarize(rows, space)
        summary["space"] = space
        summary["index"] = i
        checks.append({"summary": summary, "trajectory": rows})
        print(f"#{i} SPACE={space:.3f} frames={len(rows)} -> {summary['result']}")
    cap.release()
    report = {"file": path, "video": meta, "events": ev, "summary": {"checks": len(checks), "white": sum(x["summary"]["result"] == "WHITE" for x in checks), "black": sum(x["summary"]["result"] == "BLACK" for x in checks), "uncertain": sum(x["summary"]["result"] == "UNCERTAIN" for x in checks), "no_circle": sum(x["summary"]["result"] == "NO_CIRCLE" for x in checks)}, "checks": checks}
    report_name = Path(f"analysis_session_{Path(path).stem}.json")
    write_atomic_json(report_name, report)
    session_report = {"file": path, "report_file": report_name.name, "summary": report["summary"], "space_times": spaces, "checks": checks}
    merge_reports(session_report)
    print(f"ИТОГ: Checks={len(checks)} WHITE={report['summary']['white']} BLACK={report['summary']['black']} UNCERTAIN={report['summary']['uncertain']} NO_CIRCLE={report['summary']['no_circle']}")
    print(f"===== КОНЕЦ АНАЛИЗА {path} =====")


def eval_fraction(value):
    a, b = value.split("/")
    return float(a) / float(b)


if __name__ == "__main__":
    main()
