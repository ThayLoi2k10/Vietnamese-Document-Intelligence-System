"""One-Click Instant Demo Launcher with Cloudflare HTTPS Tunnel (Zero-Cost, Official Cloudflare Network)."""

import os
from pathlib import Path
import re
import subprocess
import sys
import time
import webbrowser

# Ensure UTF-8 output on Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent
PYTHON_EXE = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
CLOUDFLARED_EXE = PROJECT_ROOT / "cloudflared.exe"

if not PYTHON_EXE.exists():
    PYTHON_EXE = Path(sys.executable)


def log(msg: str):
    print(msg, flush=True)


def main():
    log("=" * 70)
    log("  VIETNAMESE LEGAL RAG ASSISTANT - INSTANT DEMO LAUNCHER")
    log("=" * 70)
    log("[1/3] Dang khoi dong ung dung Streamlit...")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    env["PYTHONIOENCODING"] = "utf-8"

    streamlit_proc = subprocess.Popen(
        [
            str(PYTHON_EXE),
            "-m",
            "streamlit",
            "run",
            str(PROJECT_ROOT / "app.py"),
            "--server.port=8501",
            "--server.address=0.0.0.0",
            "--server.headless=true",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
    )

    # Wait for Streamlit to start
    time.sleep(3)

    log("[2/3] Dang ket noi mang toan cau Cloudflare Tunnel...")
    if not CLOUDFLARED_EXE.exists():
        log(f"Loi: Khong tim thay file {CLOUDFLARED_EXE}")
        streamlit_proc.terminate()
        return

    tunnel_cmd = [
        str(CLOUDFLARED_EXE),
        "tunnel",
        "--url",
        "http://localhost:8501",
    ]

    tunnel_proc = subprocess.Popen(
        tunnel_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        encoding="utf-8",
        errors="ignore",
    )

    public_url = None
    start_time = time.time()

    log("[3/3] Dang thiet lap duong link HTTPS cong khai...")
    while time.time() - start_time < 20:
        line = tunnel_proc.stdout.readline()
        if not line:
            continue
        match = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", line)
        if match:
            public_url = match.group(0)
            break

    log("=" * 70)
    if public_url:
        log("🎉 TRIEN KHAI THANH CONG! HE THONG DANG HOAT DONG:")
        log(f"👉 Public URL : {public_url}")
        log(f"👉 Local URL  : http://localhost:8501")
        log("=" * 70)
        log("Trinh duyet se tu dong mo web len ngay bay gio...")
        log("Bat ky ai (ke ca tren dien thoai) deu co the mo Public URL tren de test!")
        log("Nhan Ctrl + C de dung demo khi su dung xong.")
        log("=" * 70)
        # Open in default browser automatically
        try:
            webbrowser.open(public_url)
        except Exception:
            webbrowser.open("http://localhost:8501")
    else:
        log("Chua bat duoc Public URL, ban van co the truy cap local tai:")
        log("👉 Local URL : http://localhost:8501")
        log("=" * 70)
        try:
            webbrowser.open("http://localhost:8501")
        except Exception:
            pass

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        log("\nDang tat he thong demo...")
        streamlit_proc.terminate()
        tunnel_proc.terminate()
        log("Da tat an toan!")


if __name__ == "__main__":
    main()
