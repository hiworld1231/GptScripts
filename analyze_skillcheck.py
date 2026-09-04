#!/usr/bin/env python3
import argparse
import json
import math
import os
import re
import subprocess
from itertools import product

import cv2
import numpy as np


def newest_session():
    files = [f for f in os.listdir('.') if re.fullmatch(r'session_\d+\.mkv', f)]
    if not files:
        raise FileNotFoundError('Не найден session_*.mkv в текущей папке')
    return max(files, key=lambda x: int(re.search(r'(\d+)', x).group(1)))


def ass_events(path):
    try:
        p = subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-i', path, '-map', '0:s:0', '-f', 'ass', '-'], capture_output=True, text=True)
        text = p.stdout
    except Exception:
        return []
    out = []
    for line in text.splitlines():
        if not line.startswith('Dialogue:'):
            continue
        parts = line.split(',', 9)
        if len(parts) < 10:
            continue
        m = re.match(r'(\d+):(\d+):(\d+)[.:](\d+)', parts[1])
        if not m:
            continue
        h, mi, s, cs = map(int, m.groups())
        t = h * 3600 + mi * 60 + s + cs / 100.0
        txt = parts[9].strip()
        mm = re.search(r'(LMB_DOWN|LMB_UP|SPACE_DOWN)', txt)
        if mm:
            out.append((t, mm.group(1)))
    return sorted(out)


def probe(path):
    p = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=width,height,r_frame_rate,nb_frames,duration', '-of', 'json', path], capture_output=True, text=True)
    return json.loads(p.stdout)['streams'][0]


def red_mask(frame, mode, sat, val):
    if mode == 'hsv':
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        return (((hsv[:, :, 0] <= 10) | (hsv[:, :, 0] >= 170)) & (hsv[:, :, 1] >= sat) & (hsv[:, :, 2] >= val)).astype(np.uint8)
    if mode == 'rgb':
        b, g, r = cv2.split(frame)
        return ((r > val) & (r.astype(np.int16) - g.astype(np.int16) > sat) & (r.astype(np.int16) - b.astype(np.int16) > sat)).astype(np.uint8)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    return ((lab[:, :, 1] >= val) & (lab[:, :, 1].astype(np.int16) - lab[:, :, 0].astype(np.int16) > sat)).astype(np.uint8)


def ring_hist(mask, cx, cy, r0, r1, bins=72):
    yy, xx = np.indices(mask.shape)
    dx = xx - cx
    dy = yy - cy
    r = np.sqrt(dx * dx + dy * dy)
    keep = (r >= r0) & (r <= r1)
    ang = (np.arctan2(-dy, dx) + 2 * np.pi) % (2 * np.pi)
    idx = (ang[keep] * bins / (2 * np.pi)).astype(np.int32)
    vals = mask[keep]
    hist = np.bincount(idx, weights=vals, minlength=bins).astype(np.float32)
    counts = np.bincount(idx, minlength=bins).astype(np.float32)
    return hist / np.maximum(counts, 1)


def white_hist(frame, cx, cy, r0, r1, bins=72, sat_max=70, val_min=170):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    white = ((hsv[:, :, 1] <= sat_max) & (hsv[:, :, 2] >= val_min)).astype(np.uint8)
    return ring_hist(white, cx, cy, r0, r1, bins)


def circular_stats(h):
    n = len(h)
    a = np.arange(n) * 2 * np.pi / n
    w = np.maximum(h, 0)
    z = np.sum(w * np.exp(1j * a))
    total = float(np.sum(w))
    if total <= 1e-9:
        return 0.0, 0.0
    return float((math.atan2(z.imag, z.real) + 2 * math.pi) % (2 * math.pi)), float(abs(z) / total)


def longest_run(h, threshold):
    x = h >= threshold
    if not np.any(x):
        return 0
    y = np.r_[x, x]
    best = cur = 0
    for v in y:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return min(best, len(x))


def extract(path, sample_fps=30):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    step = max(1, round(fps / sample_fps))
    rows = []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=35, param1=90, param2=28, minRadius=55, maxRadius=80)
            candidates = [] if circles is None else circles[0]
            cand = [c for c in candidates if 140 <= c[0] <= 180 and 140 <= c[1] <= 185]
            if cand:
                c = min(cand, key=lambda q: abs(q[0]-160)+abs(q[1]-163)+0.4*abs(q[2]-68))
                cx, cy, rr = map(float, c)
            else:
                cx, cy, rr = 160.0, 163.0, 68.0
            rows.append({'t': i / fps, 'frame': frame, 'cx': cx, 'cy': cy, 'r': rr})
        i += 1
    cap.release()
    return rows, fps


def build_features(rows, params):
    out = []
    for row in rows:
        f = row['frame']; cx, cy, rr = row['cx'], row['cy'], row['r']
        red = red_mask(f, params['red_mode'], params['red_sat'], params['red_val'])
        rh = ring_hist(red, cx, cy, rr + params['red_inner'], rr + params['red_outer'])
        wh = white_hist(f, cx, cy, rr + params['white_inner'], rr + params['white_outer'], sat_max=params['white_sat'], val_min=params['white_val'])
        red_ang, red_con = circular_stats(rh)
        white_ang, white_con = circular_stats(wh)
        red_mass = float(np.mean(rh))
        white_mass = float(np.mean(wh))
        white_run = longest_run(wh, max(params['white_threshold'], float(np.max(wh)) * 0.55)) / len(wh)
        score = (params['w_red'] * red_con + params['w_white'] * white_con + params['w_run'] * white_run + params['w_mass'] * red_mass)
        score *= min(1.5, max(0.3, (rr - 55) / 15))
        out.append((row['t'], score, red_ang, white_ang, red_con, white_con, white_run, cx, cy, rr))
    return out


def intervals(scores, threshold, min_duration, max_gap):
    active = np.array([s[1] >= threshold for s in scores], dtype=bool)
    times = np.array([s[0] for s in scores])
    groups=[]; start=None; last=None
    for i, on in enumerate(active):
        if on and start is None:
            start=i; last=i
        elif on:
            if times[i] - times[last] <= max_gap:
                last=i
            else:
                if times[last]-times[start] >= min_duration: groups.append((times[start], times[last]))
                start=i
            last=i
        elif start is not None and times[i]-times[last] > max_gap:
            if times[last]-times[start] >= min_duration: groups.append((times[start], times[last]))
            start=None; last=None
    if start is not None and times[last]-times[start] >= min_duration:
        groups.append((times[start], times[last]))
    return groups


def score_match(iv, spaces, tolerance=0.45):
    if not spaces:
        return 0.0, []
    matched=[]; used=set()
    for s in spaces:
        best=None
        for j,(a,b) in enumerate(iv):
            if j in used: continue
            d=0 if a <= s <= b else min(abs(s-a), abs(s-b))
            if d <= tolerance and (best is None or d < best[0]): best=(d,j)
        if best:
            used.add(best[1]); matched.append((s,best[0],iv[best[1]]))
    precision=len(matched)/max(1,len(iv)); recall=len(matched)/len(spaces)
    penalty=max(0,len(iv)-len(matched))*0.12
    return 100*(0.65*recall+0.35*precision)-100*penalty, matched


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('file', nargs='?', default=None)
    ap.add_argument('--sample-fps', type=int, default=30)
    args=ap.parse_args()
    path=args.file or newest_session()
    info=probe(path)
    events=ass_events(path)
    spaces=[t for t,k in events if k=='SPACE_DOWN']
    downs=[t for t,k in events if k=='LMB_DOWN']
    if downs:
        offset=downs[0]
        spaces=[max(0,t-offset) for t in spaces]
    print(f'Файл: {path}')
    print(f"Видео: {info.get('width')}x{info.get('height')} fps={info.get('r_frame_rate')} duration={info.get('duration')}")
    print(f'SPACE: {len(spaces)} -> ' + ', '.join(f'{x:.3f}' for x in spaces))
    print('Извлекаю кадры и геометрию...')
    rows,_=extract(path,args.sample_fps)
    print(f'Кадров для анализа: {len(rows)}')

    base={
      'red_mode':['hsv','rgb','lab'], 'red_sat':[45,70,95], 'red_val':[90,130,170],
      'red_inner':[2,5,8], 'red_outer':[9,13,18], 'white_inner':[0,2,5], 'white_outer':[4,8,12],
      'white_sat':[35,55,75], 'white_val':[150,180,205], 'white_threshold':[0.18,0.28,0.38],
      'w_red':[0.8,1.2,1.6], 'w_white':[0.6,1.0,1.4], 'w_run':[0.5,1.0,1.5], 'w_mass':[0.1,0.3,0.6]
    }
    configs=[]
    for vals in product(*base.values()):
        p=dict(zip(base,vals)); configs.append(p)
    print(f'Параметрических конфигураций: {len(configs):,}')
    print('Гоняю тесты...')
    ranked=[]
    for n,p in enumerate(configs,1):
        feats=build_features(rows,p)
        vals=np.array([x[1] for x in feats],dtype=np.float32)
        q=np.percentile(vals,[55,60,65,70,75,80,85,90,92,94,96])
        for threshold in q:
            for min_duration in (0.10,0.17,0.23,0.30):
                for gap in (0.08,0.14,0.22):
                    iv=intervals(feats,float(threshold),min_duration,gap)
                    sc,match=score_match(iv,spaces)
                    ranked.append((sc,p,threshold,min_duration,gap,iv,match))
    ranked.sort(key=lambda x:x[0],reverse=True)
    print(f'Всего тестов: {len(ranked):,}')
    print('\n========== TOP 12 ==========')
    for i,(sc,p,th,md,gap,iv,match) in enumerate(ranked[:12],1):
        print(f'#{i} score={sc:.2f} intervals={len(iv)} threshold={th:.4f} dur={md:.2f} gap={gap:.2f} mode={p["red_mode"]} red={p["red_sat"]}/{p["red_val"]} white={p["white_sat"]}/{p["white_val"]}')
        print('  intervals:', ' '.join(f'[{a:.2f},{b:.2f}]' for a,b in iv))
        print('  matches:  ', ' '.join(f'{s:.2f}->{d:.3f}' for s,d,_ in match))

    best=ranked[0]
    feats=build_features(rows,best[1])
    print('\n========== BEST CHECK DETAILS ==========')
    for s,d,(a,b) in best[6]:
        near=[x for x in feats if a-0.02<=x[0]<=b+0.02]
        if not near: continue
        k=min(near,key=lambda x:abs(x[0]-s))
        print(f'SPACE {s:.3f} interval [{a:.3f},{b:.3f}] score={k[1]:.3f} redAngle={math.degrees(k[2]):.1f}° whiteAngle={math.degrees(k[3]):.1f}° redConc={k[4]:.3f} whiteConc={k[5]:.3f} whiteRun={k[6]:.3f} center=({k[7]:.1f},{k[8]:.1f}) r={k[9]:.1f}')

    report={'file':path,'space_times':spaces,'tests':len(ranked),'top':[]}
    for sc,p,th,md,gap,iv,match in ranked[:50]:
        report['top'].append({'score':sc,'params':p,'threshold':th,'min_duration':md,'max_gap':gap,'intervals':iv,'matches':match})
    with open('analysis_report.json','w',encoding='utf-8') as f: json.dump(report,f,ensure_ascii=False,indent=2)
    print('\nОтчёт сохранён: analysis_report.json')

if __name__=='__main__':
    main()
