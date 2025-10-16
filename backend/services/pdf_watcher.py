#!/usr/bin/env python3
"""
Simple PDF watcher:
- Watches backend/input_data/pdf for new .pdf files.
- When a new file appears (created or moved in), waits for copy to finish,
  then runs: .venv/bin/python -m services.pdf_processor --src <that-file>
- Safe, single-threaded, minimal deps (requires watchdog).
"""
from pathlib import Path
import time
import subprocess
import argparse
import sys

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
except Exception:
    print("Missing dependency: pip install watchdog", file=sys.stderr)
    sys.exit(2)

BASE = Path(__file__).resolve().parents[1]
PDF_DIR = BASE / "input_data" / "pdf"
PROCESSOR_MODULE = "services.pdf_processor"

def wait_for_copy_complete(p: Path, timeout=60, interval=1.0):
    prev = -1
    elapsed = 0.0
    while elapsed < timeout:
        try:
            cur = p.stat().st_size
        except FileNotFoundError:
            return False
        if cur == prev:
            return True
        prev = cur
        time.sleep(interval)
        elapsed += interval
    return False

class Handler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory:
            return
        p = Path(event.src_path)
        if p.suffix.lower() == ".pdf":
            print("[watcher] created:", p)
            self.process(p)

    def on_moved(self, event):
        if event.is_directory:
            return
        p = Path(event.dest_path)
        if p.suffix.lower() == ".pdf":
            print("[watcher] moved:", p)
            self.process(p)

    def process(self, p: Path):
        # wait until file size stable
        ok = wait_for_copy_complete(p)
        if not ok:
            print("[watcher] copy incomplete or timeout for", p)
            return
        cmd = [str(BASE / ".venv" / "bin" / "python"), "-m", PROCESSOR_MODULE, "--src", str(p)]
        print("[watcher] running:", " ".join(cmd))
        try:
            subprocess.run(cmd, check=False)
            print("[watcher] done:", p)
        except Exception as e:
            print("[watcher] error running processor:", e)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(PDF_DIR))
    args = ap.parse_args()

    watch_dir = Path(args.dir)
    watch_dir.mkdir(parents=True, exist_ok=True)

    event_handler = Handler()
    observer = Observer()
    observer.schedule(event_handler, str(watch_dir), recursive=False)
    observer.start()
    print("[watcher] watching", watch_dir)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("[watcher] stopping")
    finally:
        observer.stop()
        observer.join()

if __name__ == "__main__":
    main()
