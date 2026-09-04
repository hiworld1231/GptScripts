#!/usr/bin/env python3
import argparse
import json
import math
import os
import re
import subprocess

import cv2
import numpy as np

REPORT_BASE = "analysis_report"
REPORT_LIMIT = 500


def newest():
    files = [x for x in os.listdir(".") if re.fullmatch(r"session_\d+\.mkv", x)]
    if not files:
        raise FileNotFoundError("Не найден session_*.mkv")
    return max(files, key=lambda x: int(re.search(r"\d+", x).group()))


def atomic_report(report):
    raw = json.dumps(report, ensure_ascii=False, separators=(",", ":"))
    chunks = []
    pos = 0
    while pos < len(raw):
        lo, hi = 1, min(len(raw) - pos, REPORT_LIMIT)
        best = 1
        while lo <= hi:
            mid = (lo + hi) // 2
            probe = json.dumps({"chunk": "", "data": raw[pos:pos + mid]}, ensure_ascii=False, separators=(",", ":"))
            if len(probe) <= REPORT_LIMIT:
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1
        chunks.append(raw[pos:pos + best])
        pos += best
    total = len(chunks)
    created = []
    for index, data in enumerate(chunks):
        n = 0
        while True:
            suffix = "" if n == 0 else f"_{n}"
            name = f"{REPORT_BASE}{suffix}.json"
            try:
                fd = os.open(name, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
                payload = json.dumps({"chunk": index + 1, "chunks": total, "data": data}, ensure_ascii=False, separators=(",", ":"))
                os.write(fd, payload.encode("utf-8"))
                os.close(fd)
                created.append(name)
                break
            except FileExistsError:
                n += 1
            except Exception:
                try:
                    os.close(fd)
                except Exception:
                    pass
                raise
    print(f"JSON: сохранено {total} частей, каждая <= {REPORT_LIMIT} символов")
    return created


def events(path):
    q = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", path, "-map", "0:s:0", "-f", "ass", "-"], capture_output=True, text=True)
    out = []
    for line in q.stdout.splitlines():
        if not line.startswith("Dialogue:"):
            continue
        z = line.split(",", 9)
        if len(z) < 10:
            continue
        m = re.match(r"(\d+):(\d+):(\d+)[.:](\d+)", z[1])
        if not m:
            continue
        h, mi, s, cs = map(int, m.groups())
        t = h * 3600 + mi * 60 + s + cs / 100
        k = re.search(r"(LMB_DOWN|LMB_UP|SPACE_DOWN)", z[9])
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
    v20, v35, v50, v65, v80 = np.percentile(vv, [20, 35, 50, 65, 80])
    s20, s50, s80 = np.percentile(ss, [20, 50, 80])
    white = np.clip((v - (v65 + 3)) / max(1, 255 - v65), 0, 1) * np.clip((s80 + 25 - s) / max(20, s80 + 25), 0, 1)
    black = np.clip((v35 - v) / max(20, v35 - v20 + 20), 0, 1) * (0.35 + 0.65 * np.clip((s + 15) / 100, 0, 1))
    red = (np.minimum(np.abs(h - 0), np.abs(h - 180)) < 16).astype(np.float32) * np.clip((s - s20) / max(20, 100 - s20), 0, 1)
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
                rows.append({
                    "t": frame_i / fps, "x": c[0], "y": c[1], "r": c[2], "score": score,
                    "white_angle": feat[0], "white_strength": feat[1], "white_peak": feat[2],
                    "black_angle": feat[3], "black_strength": feat[4], "black_peak": feat[5],
                    "red_angle": feat[6], "red_strength": feat[7], "red_peak": feat[8],
                    "edge_angle": feat[9], "edge_strength": feat[10], "edge_peak": feat[11],
                })
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


def summarize(rows, space):
    if not rows:
        return {"result": "NO_CIRCLE", "frames": 0}
    before = [x for x in rows if x["t"] <= space + 0.03]
    recent = before[-15:] if before else rows[-15:]
    red = fit_red(rows, space)
    white_angle = float(np.angle(np.mean(np.exp(1j * np.array([x["white_angle"] for x in recent])))) % (2 * np.pi)) if recent else None
    black_angle = float(np.angle(np.mean(np.exp(1j * np.array([x["black_angle"] for x in recent])))) % (2 * np.pi)) if recent else None
    if red is not None:
        wd = abs(math.degrees(circular_distance(red["predicted_angle"], white_angle))) if white_angle is not None else 999.0
        bd = abs(math.degrees(circular_distance(red["predicted_angle"], black_angle))) if black_angle is not None else 999.0
        if wd <= bd:
            result = "WHITE"
            target_distance = wd
        else:
            result = "BLACK"
            target_distance = bd
    else:
        ws = float(np.mean([x["white_strength"] for x in recent]))
        bs = float(np.mean([x["black_strength"] for x in recent]))
        result = "WHITE" if ws >= bs else "BLACK"
        target_distance = None
    return {
        "result": result,
        "frames": len(rows),
        "first_t": rows[0]["t"],
        "last_t": rows[-1]["t"],
        "center_mean": [float(np.mean([x["x"] for x in rows])), float(np.mean([x["y"] for x in rows]))],
        "radius_mean": float(np.mean([x["r"] for x in rows])),
        "white_angle_deg": math.degrees(white_angle) if white_angle is not None else None,
        "black_angle_deg": math.degrees(black_angle) if black_angle is not None else None,
        "red_prediction": red,
        "target_distance_deg": target_distance,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file", nargs="?")
    ap.add_argument("--sample-fps", type=int, default=60)
    args = ap.parse_args()
    path = args.file or newest()
    info = probe(path)
    ev = events(path)
    downs = [x[0] for x in ev if x[1] == "LMB_DOWN"]
    off = downs[0] if downs else 0.0
    spaces = [max(0.0, x[0] - off) for x in ev if x[1] == "SPACE_DOWN"]
    print(f"Файл: {path}")
    print(f"Видео: {info.get('width')}x{info.get('height')} fps={info.get('r_frame_rate')}")
    print(f"SPACE: {len(spaces)}")
    print(f"Анализ: {args.sample_fps} FPS, каждый кадр, 36 Hough-проходов, adaptive HSV, random center, WHITE > BLACK")
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    checks = []
    for index, space in enumerate(spaces, 1):
        start = max(0.0, space - 2.5)
        end = space + 0.20
        rows = local_check(cap, fps, start, end, args.sample_fps)
        result = summarize(rows, space)
        result["space"] = space
        result["index"] = index
        result["trajectory"] = rows
        checks.append(result)
        print(f"#{index} SPACE={space:.3f} frames={len(rows)} -> {result['result']}")
        if result.get("red_prediction"):
            r = result["red_prediction"]
            print(f"   RED: angle={math.degrees(r['predicted_angle']) % 360:.1f} speed={r['speed_deg_s']:.1f}deg/s residual={r['fit_residual_deg']:.2f}deg")
        if result.get("target_distance_deg") is not None:
            print(f"   TARGET DISTANCE: {result['target_distance_deg']:.2f}deg")
    cap.release()
    whites = sum(x["result"] == "WHITE" for x in checks)
    blacks = sum(x["result"] == "BLACK" for x in checks)
    report = {
        "file": path,
        "space_times": spaces,
        "sample_fps": args.sample_fps,
        "rules": {"white": "TOP 1", "black": "TOP 2", "position": "random/per-check detection", "color": "adaptive, not exact RGB"},
        "engine": {"frames": "every sampled frame", "hough_passes": 36, "window_before_space_s": 2.5, "window_after_space_s": 0.2},
        "summary": {"checks": len(checks), "white": whites, "black": blacks, "unknown": len(checks) - whites - blacks},
        "checks": checks,
    }
    atomic_report(report)
    print(f"ИТОГ: Checks={len(checks)} WHITE={whites} BLACK={blacks}")


if __name__ == "__main__":
    main()
