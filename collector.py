import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

MOUSE_DEVICE = "/dev/input/by-id/usb-_AJAZZ_2.4G_8K-event-mouse"
KEYBOARD_DEVICE = "/dev/input/by-id/usb-BY_Tech_Gaming_Keyboard-event-kbd"
REGION = "320x240+800+420"
FPS = 60
STOP_REQUESTED = False


def request_stop(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True


def log(message):
    print(message, flush=True)


def first_free_session_id():
    ids = []
    for path in Path('.').glob('session_*.mkv'):
        try:
            ids.append(int(path.stem.split('_')[-1]))
        except ValueError:
            pass
    return max(ids, default=0) + 1


def session_output(session_id):
    return Path(f"session_{session_id:04d}.mkv")


def event_worker(device_path, start_time, event_log):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    try:
        import evdev
        from evdev import ecodes
        device = evdev.InputDevice(device_path)
        with open(event_log, "a", buffering=1, encoding="utf-8") as f:
            f.write(f"{time.monotonic() - start_time:.9f}\tWORKER_READY\t{device_path}\n")
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
                elif event.code == ecodes.KEY_E:
                    if event.value == 1:
                        f.write(f"{timestamp:.9f}\tLABEL_WHITE\n")
                elif event.code == ecodes.KEY_Q:
                    if event.value == 1:
                        f.write(f"{timestamp:.9f}\tLABEL_BLACK\n")
    except Exception as exc:
        try:
            with open(event_log, "a", buffering=1, encoding="utf-8") as f:
                f.write(f"{time.monotonic() - start_time:.9f}\tWORKER_ERROR\t{device_path}\t{type(exc).__name__}: {exc}\n")
        except Exception:
            pass
        raise


def start_event_worker(device, start_time, event_log, error_log):
    error_file = open(error_log, "a", encoding="utf-8")
    process = subprocess.Popen(["sudo", "-n", sys.executable, str(Path(__file__).resolve()), "--worker", "--device", device, "--start-time", str(start_time), "--log", str(event_log)], stdin=None, stdout=subprocess.DEVNULL, stderr=error_file, preexec_fn=os.setpgrp)
    return process, error_file


def stop_process(process, sig=signal.SIGTERM, timeout=5):
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, sig)
    except (ProcessLookupError, PermissionError):
        try:
            process.send_signal(sig)
        except ProcessLookupError:
            return
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            try:
                process.kill()
            except ProcessLookupError:
                pass
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass


def parse_events(path):
    events = []
    if not path.exists():
        return events
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t", 2)
            if len(parts) < 2:
                continue
            try:
                events.append((float(parts[0]), parts[1], parts[2] if len(parts) == 3 else ""))
            except ValueError:
                continue
    return sorted(events, key=lambda x: x[0])


def ass_time(seconds):
    total = max(0, round(seconds * 1000))
    return f"{total // 3600000}:{(total % 3600000) // 60000:02d}:{(total % 60000) // 1000:02d}.{total % 1000:03d}"


def make_subtitles(events, path):
    lines = [
        "[Script Info]", "ScriptType: v4.00+", "PlayResX: 320", "PlayResY: 240", "",
        "[V4+ Styles]",
        "Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding",
        "Style: Events,DejaVu Sans,12,&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,1,0,0,0,100,100,0,0,1,2,0,8,5,5,5,1", "",
        "[Events]", "Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text"
    ]
    for timestamp, event in events:
        lines.append(f"Dialogue: 0,{ass_time(timestamp)},{ass_time(timestamp + 0.08)},Events,,0,0,0,,{timestamp:.9f}  {event}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_event_attachment(events, path):
    with open(path, "w", encoding="utf-8") as f:
        f.write("timestamp_seconds\tevent\n")
        for timestamp, event in events:
            f.write(f"{timestamp:.9f}\t{event}\n")


def analyze_output(output, log_path):
    analyzer = Path(__file__).resolve().with_name("analyze_skillcheck.py")
    if not analyzer.exists():
        log_path.write_text("analyze_skillcheck.py не найден.\n", encoding="utf-8")
        return 1
    result = subprocess.run([sys.executable, str(analyzer), str(output)], cwd=str(analyzer.parent), capture_output=True, text=True)
    text = result.stdout
    if result.stderr:
        text += ("\n" if text and not text.endswith("\n") else "") + result.stderr
    if not text:
        text = f"Анализатор завершился с кодом {result.returncode} без вывода.\n"
    log_path.write_text(text, encoding="utf-8")
    return result.returncode


def finalize_recording(raw_video, events_path, output, session_dir, analysis_log):
    try:
        events_data = json.loads(Path(events_path).read_text(encoding="utf-8"))
        events = [(float(t), str(e)) for t, e in events_data]
    except Exception as exc:
        Path(raw_video).unlink(missing_ok=True)
        Path(events_path).unlink(missing_ok=True)
        Path(analysis_log).write_text(f"Ошибка чтения событий: {exc}\n", encoding="utf-8")
        return 1
    spaces = sum(event == "SPACE_DOWN" for _, event in events)
    if spaces == 0:
        Path(raw_video).unlink(missing_ok=True)
        Path(events_path).unlink(missing_ok=True)
        Path(analysis_log).write_text("SPACE не было — файл удалён, анализ не запускался.\n", encoding="utf-8")
        return 0
    temp = Path(session_dir)
    subtitles = temp / "events.ass"
    attachment = temp / "events.tsv"
    try:
        make_subtitles(events, subtitles)
        write_event_attachment(events, attachment)
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(raw_video), "-i", str(subtitles), "-attach", str(attachment), "-map", "0", "-map", "1:0", "-c:v", "copy", "-c:a", "copy", "-c:s", "ass", "-metadata:s:s:0", "title=Collector Events", "-metadata:s:t", "mimetype=text/plain", "-metadata:s:t", "filename=events.tsv", str(output)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        Path(raw_video).unlink(missing_ok=True)
        Path(events_path).unlink(missing_ok=True)
        rc = analyze_output(output, Path(analysis_log))
        return rc
    except Exception as exc:
        Path(raw_video).unlink(missing_ok=True)
        Path(events_path).unlink(missing_ok=True)
        Path(output).unlink(missing_ok=True)
        Path(analysis_log).write_text(f"Ошибка сборки/анализа: {type(exc).__name__}: {exc}\n", encoding="utf-8")
        return 1
    finally:
        subtitles.unlink(missing_ok=True)
        attachment.unlink(missing_ok=True)
        try:
            temp.rmdir()
        except OSError:
            pass


def launch_finalizer(raw_video, events, output):
    session_dir = Path(tempfile.mkdtemp(prefix=f"violence_district_{output.stem}_"))
    events_path = session_dir / f"{output.stem}.events.json"
    analysis_log = Path(f".analysis_{output.stem}.log")
    analysis_log.write_text("Подготовка анализа...\n", encoding="utf-8")
    events_path.write_text(json.dumps(events, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--finalize", "--raw", str(raw_video), "--events", str(events_path), "--output", str(output), "--session-dir", str(session_dir), "--analysis-log", str(analysis_log)], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    return process, analysis_log


def print_analysis_log(name, analysis_log):
    if not analysis_log.exists():
        log(f"{name}: лог анализа не найден")
        return
    try:
        text = analysis_log.read_text(encoding="utf-8")
    except Exception as exc:
        log(f"{name}: не удалось прочитать лог анализа: {exc}")
        return
    log(f"\n===== АНАЛИЗ {name} =====")
    log(text.rstrip())
    log(f"===== КОНЕЦ АНАЛИЗА {name} =====\n")
    analysis_log.unlink(missing_ok=True)


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
    parser.add_argument("--session-dir")
    parser.add_argument("--analysis-log")
    args = parser.parse_args()
    if args.worker:
        event_worker(args.device, args.start_time, args.log)
        return
    if args.finalize:
        raise SystemExit(finalize_recording(Path(args.raw), Path(args.events), Path(args.output), Path(args.session_dir), Path(args.analysis_log)))
    log("=== Violence District DATA COLLECTOR ===\n")
    log(f"Область: {REGION}")
    log(f"FPS: {FPS}")
    log("Зажми ЛКМ на генераторе, проходи skill-check через Space, после каждого Space нажми E=WHITE или Q=BLACK.")
    log("Пример: Space, Space, E, Q → первый WHITE, второй BLACK.")
    log("Каждый LMB DOWN получает уникальный номер. SPACE=0 не сохраняется.")
    log("Анализ запускается отдельно для каждой сессии и не блокирует следующую запись.")
    log("Ctrl+C — остановить collector. Уже запущенные анализы не прерываются.\n")
    subprocess.run(["sudo", "-v"], check=True)
    event_root = Path(tempfile.mkdtemp(prefix="violence_district_events_"))
    event_log = event_root / "events.log"
    mouse_error = event_root / "mouse_worker.err"
    keyboard_error = event_root / "keyboard_worker.err"
    start_time = time.monotonic()
    mouse_worker, mouse_error_file = start_event_worker(MOUSE_DEVICE, start_time, event_log, mouse_error)
    keyboard_worker, keyboard_error_file = start_event_worker(KEYBOARD_DEVICE, start_time, event_log, keyboard_error)
    log("Ввод: запускаю обработчики мыши и клавиатуры...")
    time.sleep(0.15)
    if mouse_worker.poll() is None:
        log("Ввод: мышь слушается")
    else:
        log(f"Ввод: ОБРАБОТЧИК МЫШИ УПАЛ (код {mouse_worker.returncode})")
    if keyboard_worker.poll() is None:
        log("Ввод: клавиатура слушается")
    else:
        log(f"Ввод: ОБРАБОТЧИК КЛАВИАТУРЫ УПАЛ (код {keyboard_worker.returncode})")
    recording = False
    recorder = None
    session_start = None
    session_events = []
    output = None
    raw_video = None
    handled_events = 0
    next_id = first_free_session_id()
    finalizers = {}
    try:
        while not STOP_REQUESTED:
            events = parse_events(event_log)
            new_events = events[handled_events:]
            handled_events = len(events)
            for timestamp, event, extra in new_events:
                if event == "WORKER_READY":
                    log(f"[{timestamp:8.3f}s] Ввод готов: {Path(extra).name}")
                    continue
                if event == "WORKER_ERROR":
                    log(f"[{timestamp:8.3f}s] ОШИБКА обработчика: {extra}")
                    continue
                if STOP_REQUESTED:
                    break
                if event == "LMB_DOWN" and not recording:
                    recording = True
                    session_start = timestamp
                    session_events = [(0.0, event)]
                    output = session_output(next_id)
                    next_id += 1
                    raw_dir = Path(tempfile.mkdtemp(prefix=f"violence_district_capture_{output.stem}_"))
                    raw_video = raw_dir / f"capture_{output.stem}.mkv"
                    recorder = subprocess.Popen(["gpu-screen-recorder", "-w", REGION, "-f", str(FPS), "-c", "mkv", "-o", str(raw_video)], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
                    log(f"[{timestamp:8.3f}s] ЛКМ ЗАЖАТА → НАЧАЛО СБОРА: {output.name}")
                    continue
                if recording:
                    relative = max(0.0, timestamp - session_start)
                    if event != "LMB_DOWN":
                        session_events.append((relative, event))
                    if event == "SPACE_DOWN":
                        count = sum(e == "SPACE_DOWN" for _, e in session_events)
                        log(f"[{timestamp:8.3f}s] {output.name}: SPACE #{count} ({relative:.3f}s)")
                    elif event == "LABEL_WHITE":
                        log(f"[{timestamp:8.3f}s] {output.name}: E → WHITE ({relative:.3f}s)")
                    elif event == "LABEL_BLACK":
                        log(f"[{timestamp:8.3f}s] {output.name}: Q → BLACK ({relative:.3f}s)")
                    elif event == "LMB_UP":
                        log(f"[{timestamp:8.3f}s] ЛКМ ОТПУЩЕНА → ЗАВЕРШЕНИЕ: {output.name}")
                        stop_process(recorder, signal.SIGINT, timeout=10)
                        recorder = None
                        recording = False
                        spaces = sum(e == "SPACE_DOWN" for _, e in session_events)
                        if spaces == 0:
                            raw_video.unlink(missing_ok=True)
                            try:
                                raw_video.parent.rmdir()
                            except OSError:
                                pass
                            log(f"{output.name}: SPACE не нажимался → файл НЕ сохраняю")
                        else:
                            finalizer, analysis_log = launch_finalizer(raw_video, session_events, output)
                            finalizers[output.name] = (finalizer, analysis_log)
                            labels = sum(e in {"LABEL_WHITE", "LABEL_BLACK"} for _, e in session_events)
                            log(f"{output.name}: анализ запущен отдельно, SPACE={spaces}, LABELS={labels}")
                        session_start = None
                        session_events = []
                        output = None
                        raw_video = None
            for name, (process, analysis_log) in list(finalizers.items()):
                if process.poll() is not None:
                    print_analysis_log(name, analysis_log)
                    del finalizers[name]
            time.sleep(0.01)
    finally:
        if recorder is not None:
            stop_process(recorder, signal.SIGINT, timeout=10)
        stop_process(mouse_worker)
        stop_process(keyboard_worker)
        mouse_error_file.close()
        keyboard_error_file.close()
        try:
            for path in event_root.iterdir():
                path.unlink(missing_ok=True)
            event_root.rmdir()
        except OSError:
            pass
        log("Остановлено.")


if __name__ == "__main__":
    main()
