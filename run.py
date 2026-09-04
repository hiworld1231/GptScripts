#!/usr/bin/env python3
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "skillcheck_model.json"
REPORT = ROOT / "analysis_all.json"


def trainer(stop):
    while not stop.is_set():
        if REPORT.exists():
            subprocess.run([sys.executable, str(ROOT / "skillcheck_ml.py"), "--train"], cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        stop.wait(5)


def main():
    print("=== Violence District AI ===")
    print()
    print("1. MANUAL — ты играешь, Space + E/Q, AI обучается")
    print("2. AUTO   — AI анализирует новые skill-check и использует обученную модель")
    print()
    while True:
        choice = input("Выбери режим [1/2]: ").strip()
        if choice in {"1", "2"}:
            break
        print("Введи 1 или 2.")
    mode = "MANUAL" if choice == "1" else "AUTO"
    print()
    print(f"Режим: {mode}")
    if mode == "MANUAL":
        print("E = WHITE, Q = BLACK. Только E/Q считаются правильной разметкой.")
        print("AI будет автоматически переобучаться по накопленным данным.")
    else:
        if MODEL.exists():
            print("Модель найдена. Новые данные будут анализироваться с её помощью.")
        else:
            print("Модели пока нет. Сначала поиграй в MANUAL и дай ей обучиться.")
        print("AUTO сейчас не подменяет ground truth своими прогнозами.")
    print("Ctrl+C — остановить.\n")
    stop = threading.Event()
    worker = threading.Thread(target=trainer, args=(stop,), daemon=True)
    worker.start()
    env = os.environ.copy()
    env["SKILLCHECK_MODE"] = mode
    try:
        raise SystemExit(subprocess.call([sys.executable, str(ROOT / "collector.py")], cwd=ROOT, env=env))
    finally:
        stop.set()
        worker.join(timeout=1)


if __name__ == "__main__":
    main()
