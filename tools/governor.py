"""Governor: one oversight pass over everything that runs unattended. Windows runs it every 3 minutes.

What it checks, and what it may do on its own (everything else is only reported to the main agent):
  local training   the active run (checkpoints/ACTIVE_RUN): supervisor alive, log not stale, HALTED flag.
                   FIX: if the supervisor is dead and the run is not finished/paused/stopped, and the 5-minute
                   watchdog has not brought it back within 12 minutes, run `ctl.py start` once (resume).
  watchdog task    the NameThatLLM-Watchdog scheduled task exists and ran recently.
  dashboard        http://localhost:8765 answers. FIX: restart it (pythonw, --no-browser).
  local jobs       ad-hoc jobs registered in logs/governor_jobs.json (name, log, done regex): crash signatures
                   (Traceback, out of memory, Killed) or a stale log -> ALERT. Never restarted automatically.
  Kaggle kernels   status of our kernels: transitions are reported, ERROR / CANCEL -> ALERT.
  submissions      newly scored -> DONE with the score; ERROR -> ALERT; PENDING over 4 h -> ALERT.
  machine          free disk under 15 GB -> ALERT; nvidia-smi not answering -> ALERT.
It never submits, never pushes Kaggle kernels and never deletes anything.

Events go to logs/governor.log, one line each: "YYYY-mm-dd HH:MM LEVEL source: text", LEVEL in
INFO | DONE | FIXED | ALERT. The main agent streams the DONE / FIXED / ALERT lines live. Each distinct ALERT
is written once (state in logs/governor_state.json) until its condition clears.
Usage:
  .venv/Scripts/python.exe tools/governor.py              one pass (what the scheduled task runs)
  .venv/Scripts/python.exe tools/governor.py install      register the scheduled task (every 3 minutes)
  .venv/Scripts/python.exe tools/governor.py remove
  .venv/Scripts/python.exe tools/governor.py job NAME LOG [DONE_REGEX]   register an ad-hoc job to watch
  .venv/Scripts/python.exe tools/governor.py unjob NAME
"""
import csv
import io
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs"
LOGS.mkdir(exist_ok=True)
EVENTS, STATE, JOBS = LOGS / "governor.log", LOGS / "governor_state.json", LOGS / "governor_jobs.json"
PY, PYW = ROOT / ".venv/Scripts/python.exe", ROOT / ".venv/Scripts/pythonw.exe"
TASK, WATCHDOG_TASK = "NameThatLLM-Governor", "NameThatLLM-Watchdog"
KAGGLE = shutil.which("kaggle") or str(Path.home() / "AppData/Local/Programs/Python/Python311/Scripts/kaggle.exe")
KERNELS = ["varadarajborkar/ntl-qwen-train", "varadarajborkar/ntl-research", "varadarajborkar/ntl-s2-main",
           "varadarajborkar/ntl-qwen-blend"]
COMP = "name-that-llm"
CRASH = re.compile(r"Traceback|out of memory|CUDA error|Killed|MemoryError|Segmentation fault")
NOWIN = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0


def load(p, default):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return default


def save(p, obj):
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1), encoding="utf-8")
    os.replace(tmp, p)


def event(level, src, text):
    line = f"{datetime.now():%Y-%m-%d %H:%M} {level} {src}: {text}"
    with open(EVENTS, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def run(cmd, timeout=60):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, creationflags=NOWIN,
                           encoding="utf-8", errors="replace")
        return r.returncode, (r.stdout or "") + (r.stderr or "")
    except Exception as e:
        return -1, f"{type(e).__name__}: {e}"


class Alerts:
    """Each distinct alert key fires once until clear(key) is called (its condition went away)."""

    def __init__(self, st):
        self.st = st.setdefault("alerts", {})

    def raise_(self, key, src, text):
        if key not in self.st:
            self.st[key] = time.time()
            event("ALERT", src, text)

    def clear(self, key, src=None, text=None):
        if self.st.pop(key, None) is not None and src:
            event("INFO", src, text or "recovered")


def check_training(st, al):
    act = ROOT / "checkpoints/ACTIVE_RUN"
    name = act.read_text().strip() if act.exists() else ""
    if not name:
        return
    R = ROOT / "checkpoints" / name
    flags = {f for f in ("DONE_TRAINING", "PAUSE", "STOP", "HALTED") if (R / f).exists()}
    if "HALTED" in flags:
        al.raise_(f"halted:{name}", "training", f"{name} HALTED after an error; see {R / 'supervisor.log'}")
        return
    al.clear(f"halted:{name}")
    if flags:
        return
    code, out = run([str(PY), str(ROOT / "finetune/ctl.py"), "--run", name, "status"], 60)
    alive = "RUNNING" in out
    log = R / "train.log"
    quiet = time.time() - log.stat().st_mtime if log.exists() else None
    tr = st.setdefault("training", {})
    if alive:
        tr.pop(name, None)
        al.clear(f"dead:{name}", "training", f"{name} supervisor running again")
        if quiet is not None and quiet > 40 * 60:
            al.raise_(f"stale:{name}", "training", f"{name} supervisor alive but train.log silent {quiet / 60:.0f} min")
        else:
            al.clear(f"stale:{name}", "training", f"{name} log moving again")
        return
    first = tr.setdefault(name, {"dead_since": time.time(), "fixed": False})
    if time.time() - first["dead_since"] < 12 * 60:
        return  # give the 5-minute watchdog its chance first
    if not first["fixed"]:
        first["fixed"] = True
        code, out = run([str(PY), str(ROOT / "finetune/ctl.py"), "--run", name, "start"], 120)
        event("FIXED", "training", f"{name} supervisor was dead 12+ min (watchdog did not restart it): ran ctl.py "
                                   f"start -> {out.strip().splitlines()[-1] if out.strip() else code}")
    else:
        al.raise_(f"dead:{name}", "training", f"{name} still not running after one restart; needs the main agent")


def check_watchdog(al):
    code, out = run(["schtasks", "/query", "/tn", WATCHDOG_TASK, "/fo", "csv", "/v"], 30)
    if code != 0:
        al.raise_("watchdog-missing", "watchdog", f"scheduled task {WATCHDOG_TASK} not found")
        return
    al.clear("watchdog-missing", "watchdog", "task present again")
    try:
        row = next(csv.DictReader(io.StringIO(out)))
        last = row.get("Last Run Time", "")
        t = datetime.strptime(last, "%d-%m-%Y %H:%M:%S") if last and last[2] == "-" else None
    except Exception:
        t = None
    if t and (datetime.now() - t).total_seconds() > 30 * 60:
        al.raise_("watchdog-idle", "watchdog", f"{WATCHDOG_TASK} last ran {last}")
    elif t:
        al.clear("watchdog-idle", "watchdog", "running on schedule again")


def check_dashboard(st, al):
    try:
        urllib.request.urlopen("http://localhost:8765/api/runs", timeout=10).read(100)
        st["dash_fail"] = 0
        al.clear("dashboard", "dashboard", "answering again")
        return
    except Exception:
        st["dash_fail"] = st.get("dash_fail", 0) + 1
    if st["dash_fail"] == 2:  # two passes in a row (about 6 minutes): restart once
        subprocess.Popen([str(PYW), str(ROOT / "dashboard/server.py"), "--no-browser"], cwd=ROOT,
                         creationflags=NOWIN | getattr(subprocess, "DETACHED_PROCESS", 0))
        event("FIXED", "dashboard", "not answering for 2 checks: restarted on port 8765")
    elif st["dash_fail"] >= 4:
        al.raise_("dashboard", "dashboard", "still down after a restart")


def check_jobs(st, al):
    jobs = load(JOBS, {})
    for name, j in list(jobs.items()):
        log = Path(j["log"])
        if not log.is_absolute():
            log = ROOT / log
        if not log.exists():
            continue
        try:
            tail = log.read_text(encoding="utf-8", errors="replace")[-20000:]
        except Exception:
            continue
        if j.get("done") and re.search(j["done"], tail):
            event("DONE", f"job {name}", f"finished ({log.name})")
            jobs.pop(name)
            continue
        m = CRASH.search(tail)
        if m:
            snippet = tail[m.start():m.start() + 160].replace("\n", " | ")
            al.raise_(f"job:{name}", f"job {name}", f"crash signature in {log.name}: {snippet}")
            continue
        quiet = time.time() - log.stat().st_mtime
        if quiet > j.get("stale_min", 30) * 60:
            al.raise_(f"jobstale:{name}", f"job {name}", f"log silent {quiet / 60:.0f} min ({log.name})")
        else:
            al.clear(f"jobstale:{name}")
    save(JOBS, jobs)


def check_kaggle(st, al):
    ks = st.setdefault("kernels", {})
    for k in KERNELS:
        code, out = run([KAGGLE, "kernels", "status", k], 60)
        m = re.search(r"KernelWorkerStatus\.(\w+)", out)
        if not m:
            st["kaggle_fail"] = st.get("kaggle_fail", 0) + 1
            if st["kaggle_fail"] >= 5:
                al.raise_("kaggle-api", "kaggle", f"status calls failing: {out.strip()[-160:]}")
            return
        st["kaggle_fail"] = 0
        al.clear("kaggle-api", "kaggle", "API answering again")
        s, prev = m.group(1), ks.get(k)
        if s != prev:
            ks[k] = s
            if prev is not None:
                lvl = "ALERT" if s in ("ERROR", "CANCEL_REQUESTED", "CANCEL_ACKNOWLEDGED") else (
                    "DONE" if s == "COMPLETE" else "INFO")
                event(lvl, "kaggle", f"{k.split('/')[1]}: {prev} -> {s}")


def check_submissions(st, al):
    code, out = run([KAGGLE, "competitions", "submissions", "-c", COMP, "--csv"], 90)
    lines = out.splitlines()
    head = next((i for i, l in enumerate(lines) if l.startswith("ref,")), None)  # skip CLI warnings
    if head is None:
        return
    rows = [r for r in csv.DictReader(io.StringIO("\n".join(lines[head:]))) if r.get("ref", "").isdigit()]
    if not rows:
        return
    subs = st.setdefault("subs", {})
    first_pass = not subs
    for r in rows[:10]:
        ref, s, score = r["ref"], r["status"].split(".")[-1], r.get("publicScore", "")
        prev = subs.get(ref, {}).get("status")
        if prev != s:
            subs[ref] = {"status": s, "date": r["date"], "seen": time.time()}
            if not first_pass:
                lvl = {"COMPLETE": "DONE", "ERROR": "ALERT"}.get(s, "INFO")
                event(lvl, "submission", f"{ref} {s} {score} ({r['description'][:70]})")
        if s == "PENDING":
            try:
                age = (datetime.utcnow() - datetime.strptime(r["date"][:19], "%Y-%m-%d %H:%M:%S")).total_seconds()
            except Exception:
                age = 0
            if age > 4 * 3600:
                al.raise_(f"subslow:{ref}", "submission", f"{ref} PENDING for {age / 3600:.1f} h")
        else:
            al.clear(f"subslow:{ref}")


def check_machine(al):
    free = shutil.disk_usage(ROOT).free / 1e9
    if free < 15:
        al.raise_("disk", "machine", f"only {free:.1f} GB free on the project drive")
    else:
        al.clear("disk", "machine", f"disk back to {free:.0f} GB free")
    code, out = run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader"], 30)
    if code != 0:
        al.raise_("gpu", "machine", f"nvidia-smi failed: {out.strip()[-120:]}")
    else:
        al.clear("gpu", "machine", "nvidia-smi answering again")


def one_pass():
    st = load(STATE, {})
    al = Alerts(st)
    for name, fn in (("training", lambda: check_training(st, al)), ("watchdog", lambda: check_watchdog(al)),
                     ("dashboard", lambda: check_dashboard(st, al)), ("jobs", lambda: check_jobs(st, al)),
                     ("kaggle", lambda: check_kaggle(st, al)), ("submissions", lambda: check_submissions(st, al)),
                     ("machine", lambda: check_machine(al))):
        try:
            fn()
        except Exception as e:  # one broken check must not stop the others
            al.raise_(f"check:{name}", "governor", f"check '{name}' failed: {type(e).__name__}: {e}")
        else:
            al.clear(f"check:{name}")
    st["last_pass"] = datetime.now().isoformat(timespec="seconds")
    save(STATE, st)


def install():
    ps = f"""
$a = New-ScheduledTaskAction -Execute '{PYW}' -Argument '"{Path(__file__).resolve()}"' -WorkingDirectory '{ROOT}'
$t1 = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) -RepetitionInterval (New-TimeSpan -Minutes 3) -RepetitionDuration (New-TimeSpan -Days 3650)
$t2 = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$s = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 8)
Register-ScheduledTask -TaskName '{TASK}' -Action $a -Trigger @($t1, $t2) -Settings $s -Force | Out-Null
(Get-ScheduledTask -TaskName '{TASK}').State
"""
    print(run(["powershell", "-NoProfile", "-Command", ps], 120)[1].strip())


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a:
        one_pass()
    elif a[0] == "install":
        install()
    elif a[0] == "remove":
        print(run(["schtasks", "/delete", "/tn", TASK, "/f"], 30)[1].strip())
    elif a[0] == "job" and len(a) >= 3:
        jobs = load(JOBS, {})
        jobs[a[1]] = {"log": a[2], "done": a[3] if len(a) > 3 else None, "stale_min": 30}
        save(JOBS, jobs)
        print(f"watching job {a[1]}: {a[2]}")
    elif a[0] == "unjob" and len(a) == 2:
        jobs = load(JOBS, {})
        jobs.pop(a[1], None)
        save(JOBS, jobs)
    else:
        print(__doc__)
