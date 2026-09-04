import argparse
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2
import evdev
import numpy as np
from evdev import ecodes

MOUSE_DEVICE = "/dev/input/by-id/usb-_AJAZZ_2.4G_8K-event-mouse"
KEYBOARD_DEVICE = "/dev/input/by-id/usb-BY_Tech_Gaming_Keyboard-event-kbd"
REGION = "320x240+800+420"
FPS = 60
RADIUS_MIN = 45
RADIUS_MAX = 90
SAMPLE_EVERY_N_FRAMES = 2
START_CONFIRM = 3
END_MISSES = 8


def next_output():
    i = 1
    while True:
        path = Path(f"session_{i:04d}.mkv")
        if not path.exists():
            return path
        i += 1


def event_worker(device_path, start_time, event_log):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    device = evdev.InputDevice(device_path)
    with open(event_log, "a", buffering=1, encoding="utf-8") as f:
        for event in device.read_loop():
            if event.type != ecodes.EV_KEY:
                continue
            timestamp = time.monotonic() - start_time
            if event.code == ecodes.BTN_LEFT:
                if event.value == 1:
                    f.write(f"{timestamp:.9f}\tLMB_DOWN\n")
                elif event.value == 0:
                    f.write(f"{timestamp:.9f}\tLMB_UP\n")
            elif event.code == ecodes.KEY_SPACE:
                if event.value == 1:
                    f.write(f"{timestamp:.9f}\tSPACE_DOWN\n")
                elif event.value == 0:
                    f.write(f"{timestamp:.9f}\tSPACE_UP\n")


def start_event_worker(device, start_time, event_log):
    return subprocess.Popen(["sudo", sys.executable, str(Path(__file__).resolve()), "--worker", "--device", device, "--start-time", str(start_time), "--log", str(event_log)], stdin=None, stdout=None, stderr=None, preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_IGN))


def stop_process(process, sig=signal.SIGTERM, timeout=5):
    if process is None or process.poll() is not None:
        return
    try:
        process.send_signal(sig)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            process.kill()
            process.wait(timeout=2)
        except (subprocess.TimeoutExpired, ProcessLookupError):
            pass


def parse_events(path):
    events = []
    if not path.exists():
        return events
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                timestamp, event_name = line.split("\t", 1)
                events.append((float(timestamp), event_name))
            except ValueError:
                pass
    return sorted(events, key=lambda x: x[0])


def get_session_events(events, start_time, end_time):
    return [(max(0.0, timestamp - start_time), event) for timestamp, event in events if start_time <= timestamp <= end_time]


def circle_score(frame, circle):
    x, y, radius = circle
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    angles = np.linspace(0, 2 * np.pi, 96, endpoint=False)
    xs = np.rint(x + radius * np.cos(angles)).astype(np.int32)
    ys = np.rint(y + radius * np.sin(angles)).astype(np.int32)
    valid = (xs >= 2) & (xs < frame.shape[1] - 2) & (ys >= 2) & (ys < frame.shape[0] - 2)
    xs, ys = xs[valid], ys[valid]
    if len(xs) < 20:
        return -1.0
    gx = gray[ys, xs + 1].astype(np.float32) - gray[ys, xs - 1].astype(np.float32)
    gy = gray[ys + 1, xs].astype(np.float32) - gray[ys - 1, xs].astype(np.float32)
    return float(np.percentile(np.sqrt(gx * gx + gy * gy), 75))


def find_skill_circle(frame):
    gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 1.2)
    candidates = []
    for p2 in (21, 24, 27, 30, 33):
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.15, minDist=35, param1=90, param2=p2, minRadius=RADIUS_MIN, maxRadius=RADIUS_MAX)
        if circles is not None:
            candidates.extend((float(x), float(y), float(r)) for x, y, r in circles[0])
    if not candidates:
        return None
    unique = []
    for c in candidates:
        if not any(np.hypot(c[0] - q[0], c[1] - q[1]) < 8 and abs(c[2] - q[2]) < 8 for q in unique):
            unique.append(c)
    return max(unique, key=lambda c: circle_score(frame, c))


def detect_skill_checks(video_path):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return []
    fps = capture.get(cv2.CAP_PROP_FPS) or FPS
    detections = []
    active = False
    start_frame = None
    last_hit_frame = None
    confirm = 0
    misses = 0
    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index % SAMPLE_EVERY_N_FRAMES:
            frame_index += 1
            continue
        circle = find_skill_circle(frame)
        if circle is not None:
            confirm += 1
            misses = 0
            last_hit_frame = frame_index
            if not active and confirm >= START_CONFIRM:
                active = True
                start_frame = frame_index - (START_CONFIRM - 1) * SAMPLE_EVERY_N_FRAMES
        else:
            confirm = 0
            if active:
                misses += 1
                if misses >= END_MISSES:
                    detections.append({"start": max(0.0, start_frame / fps), "end": max(0.0, (last_hit_frame or frame_index) / fps)})
                    active = False
                    start_frame = None
                    last_hit_frame = None
                    misses = 0
        frame_index += 1
    if active and start_frame is not None:
        detections.append({"start": start_frame / fps, "end": (last_hit_frame or frame_index) / fps})
    capture.release()
    merged = []
    for item in detections:
        if not merged or item["start"] - merged[-1]["end"] > 0.25:
            merged.append(item)
        else:
            merged[-1]["end"] = max(merged[-1]["end"], item["end"])
    return merged


def ass_time(seconds):
    total = max(0, round(seconds * 1000))
    return f"{total // 3600000}:{(total % 3600000) // 60000:02d}:{(total % 60000) // 1000:02d}.{total % 1000:03d}"


def make_subtitles(events, skill_checks, path):
    items = [(t, t + 0.08, f"{t:.9f}  {e}") for t, e in events]
    items += [(x["start"], x["end"], f"SKILL_CHECK_{i}") for i, x in enumerate(skill_checks, 1)]
    items.sort(key=lambda x: x[0])
    lines = ["[Script Info]", "ScriptType: v4.00+", "PlayResX: 320", "PlayResY: 240", "", "[V4+ Styles]", "Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding", "Style: Events,DejaVu Sans,12,&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,1,0,0,0,100,100,0,0,1,2,0,8,5,5,5,1", "", "[Events]", "Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text"]
    for start, end, text in items:
        lines.append(f"Dialogue: 0,{ass_time(start)},{ass_time(end)},Events,,0,0,0,,{text}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_event_attachment(events, path):
    with open(path, "w", encoding="utf-8") as f:
        f.write("timestamp_seconds\tevent\n")
        for timestamp, event in events:
            f.write(f"{timestamp:.9f}\t{event}\n")


def analyze_output(output):
    analyzer = Path(__file__).resolve().with_name("analyze_skillcheck.py")
    if not analyzer.exists():
        print("Анализатор не найден: analyze_skillcheck.py")
        return
    print("\n========== АВТОАНАЛИЗ ==========")
    subprocess.run([sys.executable, str(analyzer), str(output)], cwd=str(analyzer.parent))


def process_recording(raw_video, events, session_start, session_end, output):
    normalized_events = get_session_events(events, session_start, session_end)
    space_down = sum(event == "SPACE_DOWN" for _, event in normalized_events)
    if space_down == 0:
        print("ЛКМ отпущена, но SPACE не нажимался → файл НЕ сохраняю.")
        try:
            raw_video.unlink()
        except FileNotFoundError:
            pass
        return False
    with tempfile.TemporaryDirectory(prefix="violence_district_session_") as temp_dir:
        temp = Path(temp_dir)
        event_attachment = temp / "events.tsv"
        subtitles = temp / "events.ass"
        skill_checks = detect_skill_checks(raw_video)
        print(f"Найдено skill-check: {len(skill_checks)}")
        for index, item in enumerate(skill_checks, 1):
            print(f"  #{index}: {item['start']:.3f}s - {item['end']:.3f}s")
        write_event_attachment(normalized_events, event_attachment)
        make_subtitles(normalized_events, skill_checks, subtitles)
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(raw_video), "-i", str(subtitles), "-attach", str(event_attachment), "-map", "0", "-map", "1:0", "-c:v", "copy", "-c:a", "copy", "-c:s", "ass", "-metadata:s:s:0", "title=Collector Events", "-metadata:s:t", "mimetype=text/plain", "-metadata:s:t", "filename=events.tsv", str(output)], check=True)
    try:
        raw_video.unlink()
    except FileNotFoundError:
        pass
    lmb_down_count = sum(event == "LMB_DOWN" for _, event in normalized_events)
    lmb_up_count = sum(event == "LMB_UP" for _, event in normalized_events)
    print("\n========== ГОТОВО ==========")
    print(f"Файл: {output}")
    print(f"LMB DOWN: {lmb_down_count}")
    print(f"LMB UP: {lmb_up_count}")
    print(f"SPACE DOWN: {space_down}")
    print(f"SKILL-CHECK: {len(skill_checks)}")
    print(f"Размер: {output.stat().st_size / 1024 / 1024:.2f} MB")
    print("============================")
    analyze_output(output)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--device")
    parser.add_argument("--start-time", type=float)
    parser.add_argument("--log")
    args = parser.parse_args()
    if args.worker:
        event_worker(args.device, args.start_time, args.log)
        return
    print("=== Violence District DATA COLLECTOR ===\n")
    print(f"Область: {REGION}")
    print(f"FPS: {FPS}\n")
    print("Зажми ЛКМ на генераторе.")
    print("Проходи skill-check вручную через Space.")
    print("Отпусти ЛКМ после теста.\n")
    print("Каждый новый LMB DOWN создаёт новый session_XXXX.mkv.")
    print("Если SPACE не нажимался, сессия не сохраняется.")
    print("После сохранения каждого файла автоматически запускается анализ.\n")
    print("Ожидаю ЛКМ...\n")
    subprocess.run(["sudo", "-v"], check=True)
    with tempfile.TemporaryDirectory(prefix="violence_district_") as temp_dir:
        temp = Path(temp_dir)
        event_log = temp / "events.log"
        start_time = time.monotonic()
        mouse_worker = start_event_worker(MOUSE_DEVICE, start_time, event_log)
        keyboard_worker = start_event_worker(KEYBOARD_DEVICE, start_time, event_log)
        recording = False
        recorder = None
        session_start = None
        output = None
        raw_video = None
        try:
            while True:
                events = parse_events(event_log)
                lmb_state = False
                current_down = None
                current_up = None
                for timestamp, event in events:
                    if event == "LMB_DOWN":
                        lmb_state = True
                        current_down = timestamp
                        current_up = None
                    elif event == "LMB_UP":
                        if lmb_state:
                            current_up = timestamp
                        lmb_state = False
                if not recording and lmb_state and current_down is not None:
                    recording = True
                    session_start = current_down
                    output = next_output()
                    raw_video = temp / f"capture_{output.stem}.mkv"
                    print(f"\nЛКМ зажата → НАЧАЛО СБОРА: {output.name}\n")
                    recorder = subprocess.Popen(["gpu-screen-recorder", "-w", REGION, "-f", str(FPS), "-c", "mkv", "-o", str(raw_video)], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_IGN))
                if recording and current_up is not None and current_up >= session_start:
                    print("\nЛКМ отпущена → ОСТАНОВКА\n")
                    session_end = current_up
                    stop_process(recorder, signal.SIGINT, 8)
                    recorder = None
                    recording = False
                    process_recording(raw_video, events, session_start, session_end, output)
                    session_start = None
                    output = None
                    raw_video = None
                    print("\nОжидаю ЛКМ...\n")
                if recorder is not None and recorder.poll() is not None:
                    print("ОШИБКА: gpu-screen-recorder завершился во время записи.")
                    recorder = None
                    recording = False
                    session_start = None
                    output = None
                    raw_video = None
                time.sleep(0.01)
        except KeyboardInterrupt:
            print("\nОстанавливаю запись...")
            stop_process(recorder, signal.SIGINT, 5)
        finally:
            stop_process(recorder, signal.SIGINT, 5)
            stop_process(mouse_worker, signal.SIGTERM, 2)
            stop_process(keyboard_worker, signal.SIGTERM, 2)


if __name__ == "__main__":
    main()
