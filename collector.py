import argparse
import json
import os
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
STOP_REQUESTED = False


def request_stop(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True


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
    return subprocess.Popen(
        ["sudo", sys.executable, str(Path(__file__).resolve()), "--worker", "--device", device, "--start-time", str(start_time), "--log", str(event_log)],
        stdin=None,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_IGN),
    )


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
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    candidates = []
    for blur_size, sigma in ((5, 1.2), (7, 1.5), (9, 2.0)):
        blurred = cv2.GaussianBlur(gray, (blur_size, blur_size), sigma)
        for dp, p1, p2 in ((1.05, 85, 20), (1.15, 90, 24), (1.25, 100, 28), (1.35, 110, 32), (1.45, 120, 36)):
            circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=dp, minDist=30, param1=p1, param2=p2, minRadius=RADIUS_MIN, maxRadius=RADIUS_MAX)
            if circles is not None:
                candidates.extend((float(x), float(y), float(r)) for x, y, r in circles[0])
    if not candidates:
        return None
    unique = []
    for c in candidates:
        if not any(np.hypot(c[0] - q[0], c[1] - q[1]) < 7 and abs(c[2] - q[2]) < 7 for q in unique):
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
    lines = [
        "[Script Info]", "ScriptType: v4.00+", "PlayResX: 320", "PlayResY: 240", "",
        "[V4+ Styles]",
        "Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding",
        "Style: Events,DejaVu Sans,12,&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,1,0,0,0,100,100,0,0,1,2,0,8,5,5,5,1", "",
        "[Events]", "Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text"
    ]
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
    print(f"\n[{output.name}] АВТОАНАЛИЗ")
    result = subprocess.run([sys.executable, str(analyzer), str(output)], cwd=str(analyzer.parent))
    if result.returncode != 0:
        print(f"[{output.name}] Анализатор завершился с кодом {result.returncode}")


def finalize_recording(raw_video, events_path, output):
    try:
        events_data = json.loads(Path(events_path).read_text(encoding="utf-8"))
        events = [(float(t), str(e)) for t, e in events_data]
    except Exception as exc:
        print(f"Ошибка чтения событий для {output.name}: {exc}")
        Path(raw_video).unlink(missing_ok=True)
        Path(events_path).unlink(missing_ok=True)
        return 1

    space_down = sum(event == "SPACE_DOWN" for _, event in events)
    if space_down == 0:
        print(f"{output.name}: SPACE не нажимался → файл НЕ сохраняю.")
        Path(raw_video).unlink(missing_ok=True)
        Path(events_path).unlink(missing_ok=True)
        return 0

    try:
        with tempfile.TemporaryDirectory(prefix="violence_district_session_") as temp_dir:
            temp = Path(temp_dir)
            event_attachment = temp / "events.tsv"
            subtitles = temp / "events.ass"
            print(f"\n[{output.name}] Анализирую skill-check...")
            skill_checks = detect_skill_checks(Path(raw_video))
            print(f"[{output.name}] Найдено skill-check: {len(skill_checks)}")
            for index, item in enumerate(skill_checks, 1):
                print(f"[{output.name}]   #{index}: {item['start']:.3f}s - {item['end']:.3f}s")
            write_event_attachment(events, event_attachment)
            make_subtitles(events, skill_checks, subtitles)
            subprocess.run([
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-i", str(raw_video), "-i", str(subtitles), "-attach", str(event_attachment),
                "-map", "0", "-map", "1:0", "-c:v", "copy", "-c:a", "copy", "-c:s", "ass",
                "-metadata:s:s:0", "title=Collector Events", "-metadata:s:t", "mimetype=text/plain",
                "-metadata:s:t", "filename=events.tsv", str(output)
            ], check=True)
        Path(raw_video).unlink(missing_ok=True)
        Path(events_path).unlink(missing_ok=True)
        print(f"\n[{output.name}] ГОТОВО: SPACE={space_down}, SKILL-CHECK={len(skill_checks)}, размер={output.stat().st_size / 1024 / 1024:.2f} MB")
        analyze_output(output)
        return 0
    except Exception as exc:
        print(f"[{output.name}] ОШИБКА обработки: {exc}")
        return 1


def launch_finalizer(raw_video, events, output, temp_dir):
    events_path = temp_dir / f"{output.stem}.events.json"
    events_path.write_text(json.dumps(events, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve()), "--finalize", "--raw", str(raw_video), "--events", str(events_path), "--output", str(output)],
        stdin=subprocess.DEVNULL,
        stdout=None,
        stderr=None,
        start_new_session=True,
    )


def main():
    global STOP_REQUESTED
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument("--device")
    parser.add_argument("--start-time", type=float)
    parser.add_argument("--log")
    parser.add_argument("--raw")
    parser.add_argument("--events")
    parser.add_argument("--output")
    args = parser.parse_args()

    if args.worker:
        event_worker(args.device, args.start_time, args.log)
        return
    if args.finalize:
        raise SystemExit(finalize_recording(Path(args.raw), Path(args.events), Path(args.output)))

    print("=== Violence District DATA COLLECTOR ===\n")
    print(f"Область: {REGION}")
    print(f"FPS: {FPS}\n")
    print("Зажми ЛКМ на генераторе.")
    print("Проходи skill-check вручную через Space.")
    print("Отпусти ЛКМ после теста.\n")
    print("Каждый новый LMB DOWN создаёт отдельный session_XXXX.mkv.")
    print("Если SPACE не нажимался, сессия не сохраняется.")
    print("Анализ каждого завершённого файла работает отдельно и не блокирует следующую запись.\n")
    print("Ctrl+C — остановить collector. Фоновые анализы продолжат работу.\n")
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
        session_events = []
        handled_events = 0
        finalizers = []

        try:
            while not STOP_REQUESTED:
                events = parse_events(event_log)
                new_events = events[handled_events:]
                handled_events = len(events)

                for timestamp, event in new_events:
                    if STOP_REQUESTED:
                        break
                    if event == "LMB_DOWN" and not recording:
                        recording = True
                        session_start = timestamp
                        session_events = [(timestamp, event)]
                        output = next_output()
                        raw_video = temp / f"capture_{output.stem}.mkv"
                        recorder = subprocess.Popen(
                            ["gpu-screen-recorder", "-w", REGION, "-f", str(FPS), "-c", "mkv", "-o", str(raw_video)],
                            stdin=subprocess.DEVNULL,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                            start_new_session=True,
                        )
                        print(f"\nЛКМ зажата → НАЧАЛО СБОРА: {output.name}\n")
                    elif recording:
                        session_events.append((timestamp, event))
                        if event == "LMB_UP":
                            session_end = timestamp
                            print(f"\nЛКМ отпущена → {output.name}: запись завершена, анализ отправлен в фон\n")
                            stop_process(recorder, signal.SIGINT, 8)
                            recorder = None
                            recording = False
                            normalized = get_session_events(session_events, session_start, session_end)
                            finalizers.append(launch_finalizer(raw_video, normalized, output, temp))
                            session_start = None
                            session_events = []
                            print("Ожидаю ЛКМ...\n")

                if recorder is not None and recorder.poll() is not None:
                    print("ОШИБКА: gpu-screen-recorder завершился во время записи.")
                    recorder = None
                    recording = False
                    session_start = None
                    session_events = []

                finalizers = [p for p in finalizers if p.poll() is None]
                time.sleep(0.01)
        finally:
            if STOP_REQUESTED:
                print("\nОстанавливаю collector...")
            stop_process(recorder, signal.SIGINT, 5)
            stop_process(mouse_worker, signal.SIGTERM, 2)
            stop_process(keyboard_worker, signal.SIGTERM, 2)
            if finalizers:
                print(f"Фоновых анализов продолжают работу: {len(finalizers)}")


if __name__ == "__main__":
    main()
