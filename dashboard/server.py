"""Local training dashboard. Standard library only, works offline, listens on localhost only.

  .venv\\Scripts\\python.exe dashboard\\server.py            # then open http://localhost:8765
  .venv\\Scripts\\python.exe dashboard\\server.py --port 9000 --run run1 --no-browser

The page reads the run's log files and the GPU. Its Start/Pause/Stop buttons call finetune/ctl.py, the same
commands you can type yourself. Closing this server does NOT affect training.
"""
import argparse
import json
import math
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time
import webbrowser
from collections import deque
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
CTL = str(ROOT / "finetune" / "ctl.py")
NO_WINDOW = 0x08000000

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8765)
ap.add_argument("--run", default="", help="default: the most recently started run")
ap.add_argument("--no-browser", action="store_true")
args = ap.parse_args()
if not args.run:
    act = ROOT / "checkpoints" / "ACTIVE_RUN"
    args.run = act.read_text().strip() if act.exists() and act.read_text().strip() else "run1"
RUN = ROOT / "checkpoints" / args.run

STEP_RE = re.compile(r"^(\d\d):(\d\d):(\d\d) step (\d+)/(\d+) loss ([\d.]+) lr ([\d.eE+-]+) eta ([\d.]+)h")
PLAN_RE = re.compile(r"plan: (\d+) steps/epoch x (\d+) = (\d+) optimizer steps, effective batch (\d+)")
KEY_RE = re.compile(r"^(\d\d:\d\d:\d\d) (EVAL .*|resumed from .*|pause requested.*|FATAL.*|FINAL .*|start at step .*)$")
EVAL_EVERY_SECONDS = 120   # measured cost of one periodic eval on 900 passages
FINAL_EVAL_SECONDS = 360   # measured cost of the final full-validation eval
HANG_SECONDS = 30 * 60

# ------------------------------------------------------------------ GPU sampling
gpu_hist = deque(maxlen=720)  # one sample per 5 s = the last hour while this server runs
gpu_now = {}


def poll_gpu():
    q = "utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,clocks.sm,clocks.max.sm"
    while True:
        try:
            out = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=10, creationflags=NO_WINDOW).stdout
            v = [float(x) if x.strip() not in ("", "[N/A]") else None for x in out.strip().split(",")]
            gpu_now.update(util=v[0], mem=v[1], mem_total=v[2], temp=v[3], power=v[4], clock=v[5], clock_max=v[6])
            gpu_hist.append({"t": time.time(), **gpu_now})
        except Exception as e:  # nvidia-smi missing or busy: keep serving the rest
            gpu_now["error"] = str(e)
        time.sleep(5)


# ------------------------------------------------------------------ cached slow checks
_cache = {}


def cached(key, ttl, fn):
    now = time.time()
    if key not in _cache or now - _cache[key][0] > ttl:
        try:
            _cache[key] = (now, fn())
        except Exception:
            _cache[key] = (now, _cache.get(key, (0, None))[1])
    return _cache[key][1]


def supervisor_alive():
    f = RUN / "supervisor.pid"
    pid = int(f.read_text()) if f.exists() and f.read_text().strip().isdigit() else 0
    if not pid:
        return False
    ps = f"(Get-CimInstance Win32_Process -Filter 'ProcessId={pid}').CommandLine"
    out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True,
                         creationflags=NO_WINDOW).stdout
    return "_supervise" in out and "ctl.py" in out


def watchdog_state():
    out = subprocess.run(["schtasks", "/query", "/tn", "NameThatLLM-Watchdog", "/fo", "csv", "/nh"],
                         capture_output=True, text=True, creationflags=NO_WINDOW)
    return out.stdout.strip().split(",")[-1].strip('"') if out.returncode == 0 else "not installed"


def ckpt_size_gb():
    d = RUN
    return round(sum(p.stat().st_size for p in d.rglob("*") if p.is_file()) / 2**30, 2) if d.exists() else 0


# ------------------------------------------------------------------ parsing
def read_lines(p, n=None):
    try:
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    except FileNotFoundError:
        return []
    return lines[-n:] if n else lines


def parse_train_log():
    pts, speeds, events = [], [], []
    bs, total, secs_prev, step_prev, day, active = 32, 0, None, None, 0, 0.0
    for line in read_lines(RUN / "train.log"):
        m = PLAN_RE.search(line)
        if m:
            total, bs = int(m.group(3)), int(m.group(4))
        k = KEY_RE.match(line)
        if k and not k.group(2).startswith("checkpoint"):
            events.append({"time": k.group(1), "text": k.group(2)})
        m = STEP_RE.match(line)
        if not m:
            continue
        h, mi, s, step, tot, loss, lr, eta = m.groups()
        secs = int(h) * 3600 + int(mi) * 60 + int(s) + day * 86400
        if secs_prev is not None and secs < secs_prev:
            day += 1
            secs += 86400
        step, total = int(step), int(tot)
        if secs_prev is not None and step_prev is not None and step > step_prev and 0 < secs - secs_prev <= 900:
            speeds.append({"step": step, "v": round((step - step_prev) * bs / (secs - secs_prev), 3)})
            active += secs - secs_prev
        pts.append({"step": step, "loss": float(loss), "lr": float(lr), "eta": float(eta)})
        secs_prev, step_prev = secs, step
    return pts, speeds, events[-14:], bs, total, active


def parse_metrics():
    out, last_rec = [], None
    for line in read_lines(RUN / "metrics.jsonl"):
        try:
            r = json.loads(line)
            out.append({"step": r["step"], "acc": r["val_bal_acc"], "acc_bal": r.get("val_bal_acc_balanced")})
            if r.get("recall") and r.get("confusion"):
                last_rec = r
        except Exception:
            pass
    return out, last_rec


def recall_info(last_rec, balanced=False):
    """Per-model recall for the newest eval, plus who each model is most often mistaken for."""
    src, step = "subset of 900 passages", None
    rec = cm = None
    if balanced:
        if not last_rec or not last_rec.get("recall_balanced"):
            return None
        last_rec = dict(last_rec, recall=last_rec["recall_balanced"], confusion=last_rec["confusion_balanced"])
    try:
        if balanced:
            raise FileNotFoundError  # the final file holds raw results only
        fin = json.loads((RUN / "final_eval.json").read_text())
        rec, cm, src, step = fin["recall"], fin["confusion"], "full validation set (2,699 passages)", "final"
    except Exception:
        if last_rec:
            rec, cm, step = last_rec["recall"], last_rec["confusion"], last_rec["step"]
    try:
        classes = json.loads((RUN / "classes.json").read_text())
    except Exception:
        classes = []
    if not rec or len(classes) != len(rec):
        return None
    rows = []
    for i, c in enumerate(classes):
        total = sum(cm[i]) or 1
        j = max((k for k in range(len(classes)) if k != i), key=lambda k: cm[i][k])
        rows.append({"model": c, "recall": rec[i], "n": total, "confused_with": classes[j], "confused_share": round(cm[i][j] / total, 4)})
    return {"step": step, "source": src, "rows": rows}


def system_events():
    ev = []
    for name in ("supervisor.log", "watchdog.log"):
        for line in read_lines(RUN / name, 40):
            if len(line) > 20:
                ev.append({"time": line[:19], "text": line[20:], "src": name.split(".")[0]})
    ev.sort(key=lambda e: e["time"])
    return ev[-14:]


def run_setup():
    """What this run trains on, read from its own log (so the page never shows another run's setup)."""
    txt = "\n".join(read_lines(RUN / "train.log")[:40])
    n = re.search(r"data: train ([\d,]+)", txt)
    ms = re.search(r"multi-scale: (\d+) layers", txt)
    arch = f"first {ms.group(1)} layers, multi-depth pooling" if ms else "all layers, last-token head"
    return f"Qwen3-4B with QLoRA ({arch}) · {n.group(1) if n else '?'} training passages"


def build_state():
    pts, speeds, events, bs, total, active = parse_train_log()
    evals, last_rec = parse_metrics()
    prog = {}
    try:
        prog = json.loads((RUN / "progress.json").read_text())
    except Exception:
        pass
    flag = lambda n: (RUN / n).exists()
    alive = cached("alive", 15, supervisor_alive)
    log_f = RUN / "train.log"
    quiet = time.time() - log_f.stat().st_mtime if log_f.exists() else None
    step = pts[-1]["step"] if pts else 0
    total = total or prog.get("total") or 0

    if flag("DONE_TRAINING"):
        status, label = "finished", "Finished"
    elif flag("HALTED"):
        status, label = "halted", "Halted after an error"
    elif flag("STOP"):
        status, label = "stopped", "Stopped"
    elif flag("PAUSE"):
        status, label = ("pausing", "Pausing, saving a checkpoint") if alive else ("paused", "Paused")
    elif alive and quiet is not None and quiet > HANG_SECONDS:
        status, label = "stalled", "No progress for 30+ minutes"
    elif alive:
        status, label = "running", "Running"
    elif not RUN.exists() or not log_f.exists():
        status, label = "idle", "Not started"
    else:
        status, label = "down", "Not running (watchdog restarts it within 5 minutes)"

    recent = [s["v"] for s in speeds[-12:]]
    speed = statistics.median(recent) if recent else None
    eta_s = finish = None
    if speed and total and status not in ("finished",):
        left = max(total - step, 0)
        evals_left = max(math.ceil(total / 250) - len(evals), 0)
        eta_s = left * bs / speed + evals_left * EVAL_EVERY_SECONDS + FINAL_EVAL_SECONDS
        finish = (datetime.now() + timedelta(seconds=eta_s)).strftime("%a %d %b, %H:%M")

    du = shutil.disk_usage(ROOT)
    return {
        "run": args.run, "setup": run_setup(), "status": status, "label": label, "now": time.time(),
        "step": step, "total": total, "batch": bs, "samples_done": step * bs,
        "active_seconds": active, "speed": speed, "eta_seconds": eta_s, "finish": finish,
        "last_checkpoint": prog.get("step", 0), "best_acc": prog.get("best", 0),
        "has_checkpoint": any((RUN / d / "OK").exists() for d in ("ckpt_latest", "ckpt_prev", "ckpt_tmp")),
        "loss": pts, "evals": evals, "speed_pts": speeds, "recall": recall_info(last_rec), "recall_bal": recall_info(last_rec, balanced=True),
        "gpu": dict(gpu_now), "gpu_hist": list(gpu_hist)[-360:],
        "disk_free_gb": round(du.free / 2**30, 1), "run_dir_gb": cached("size", 60, ckpt_size_gb),
        "watchdog": cached("wd", 60, watchdog_state),
        "train_events": events, "system_events": system_events(),
        "log_tail": read_lines(RUN / "train.log", 30),
        "halted_reason": (RUN / "HALTED").read_text() if flag("HALTED") else "",
    }


# ------------------------------------------------------------------ actions
def run_ctl(cmd):
    out = subprocess.run([PY, CTL, "--run", args.run, cmd], capture_output=True, text=True, timeout=90,
                         creationflags=NO_WINDOW, cwd=ROOT)
    _cache.pop("alive", None)
    return {"ok": out.returncode == 0, "message": (out.stdout or out.stderr).strip()[:400]}


ACTIONS = {"start": "start", "pause": "pause", "stop": "kill"}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _ok_host(self):
        # blocks DNS-rebinding: only our own localhost names are accepted
        return self.headers.get("Host", "") in (f"localhost:{args.port}", f"127.0.0.1:{args.port}")

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype + "; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if not self._ok_host():
            return self._send(403, '{"error":"bad host"}')
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            return self._send(200, (HERE / "index.html").read_bytes(), "text/html")
        if path == "/api/state":
            return self._send(200, json.dumps(build_state()))
        self._send(404, '{"error":"not found"}')

    def do_POST(self):
        # custom header + JSON body: a web page on another origin cannot send these without our consent
        if self.path.split("?")[0] != "/api/action" or not self._ok_host() or self.headers.get("X-Requested-With") != "dashboard":
            return self._send(403, '{"error":"forbidden"}')
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            cmd = ACTIONS[body["action"]]
        except Exception:
            return self._send(400, '{"error":"bad request"}')
        self._send(200, json.dumps(run_ctl(cmd)))


if __name__ == "__main__":
    threading.Thread(target=poll_gpu, daemon=True).start()
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://localhost:{args.port}"
    print(f"Dashboard for '{args.run}' at {url}   (Ctrl+C to stop; training is not affected)")
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        sys.exit(0)
