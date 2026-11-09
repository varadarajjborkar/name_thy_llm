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
  .venv/Scripts/python.exe tools/governor.py job NAME LOG DONE_REGEX     register an ad-hoc job to watch
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
CRASH = re.compile(r"Traceback \(most recent call last\)|CUDA out of memory|CUDA error|^Killed\b|MemoryError|"
                   r"Segmentation fault", re.M)
NOWIN = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
LOCK = LOGS / "governor.lock"


def wmi_start(cmdline):
    """Start a process through WMI so it lives outside the caller's (Task Scheduler / terminal) job object."""
    cmd = cmdline.replace("'", "''")
    ps = ("$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments "
          f"@{{CommandLine='{cmd}'; CurrentDirectory='{ROOT}'}}; $r.ReturnValue")
    return run(["powershell", "-NoProfile", "-Command", ps], 60)


def load(p, default):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return default


def save(p, obj):
    tmp = p.with_suffix(f".{os.getpid()}.tmp")
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
    tr = st.setdefault("training", {})
    if "HALTED" in flags:
        al.raise_(f"halted:{name}", "training", f"{name} HALTED after an error; see {R / 'supervisor.log'}")
        return
    al.clear(f"halted:{name}")
    if flags:  # finished, paused or stopped on purpose: nothing to watch, forget old state
        tr.pop(name, None)
        al.clear(f"dead:{name}")
        al.clear(f"stale:{name}")
        return
    code, out = run([str(PY), str(ROOT / "finetune/ctl.py"), "--run", name, "status"], 60)
    alive = "supervisor RUNNING" in out
    log = R / "train.log"
    quiet = time.time() - log.stat().st_mtime if log.exists() else None
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
    if code == -1:
        al.raise_("watchdog-query", "watchdog", f"schtasks query failed: {out.strip()[-120:]}")
        return
    al.clear("watchdog-query")
    if code != 0:
        al.raise_("watchdog-missing", "watchdog", f"scheduled task {WATCHDOG_TASK} not found")
        return
    al.clear("watchdog-missing", "watchdog", "task present again")
    last, t = "", None
    try:
        last = next(csv.DictReader(io.StringIO(out))).get("Last Run Time", "")
        for fmt in ("%d-%m-%Y %H:%M:%S", "%d/%m/%Y %H:%M:%S", "%m/%d/%Y %I:%M:%S %p", "%Y-%m-%d %H:%M:%S"):
            try:
                t = datetime.strptime(last, fmt)
                break
            except ValueError:
                pass
    except Exception:
        pass
    if t is None:
        al.raise_("watchdog-parse", "watchdog", f"cannot read the task's last run time ({last!r})")
        return
    al.clear("watchdog-parse")
    if (datetime.now() - t).total_seconds() > 30 * 60:
        al.raise_("watchdog-idle", "watchdog", f"{WATCHDOG_TASK} last ran {last}")
    else:
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
        code, out = wmi_start(f'"{PYW}" "{ROOT / "dashboard/server.py"}" --no-browser')
        if out.strip().splitlines()[-1:] == ["0"]:
            event("FIXED", "dashboard", "not answering for 2 checks: restart launched (the next pass confirms it)")
        else:
            al.raise_("dashboard", "dashboard", f"restart could not be launched: {out.strip()[-120:]}")
    elif st["dash_fail"] >= 4:
        al.raise_("dashboard", "dashboard", "still down after a restart")


def clear_job(al, name):
    for k in (f"job:{name}", f"jobstale:{name}", f"jobmissing:{name}"):
        al.clear(k)


def check_jobs(st, al):
    finished, jobs_now = [], load(JOBS, {})
    for name, j in jobs_now.items():
        log = Path(j["log"])
        if not log.is_absolute():
            log = ROOT / log
        if not log.exists():
            if time.time() - j.get("registered", time.time()) > 10 * 60:
                al.raise_(f"jobmissing:{name}", f"job {name}", f"log {log} still missing 10 min after registering")
            continue
        al.clear(f"jobmissing:{name}")
        try:  # only what the job wrote after it was registered (a reused log's old text does not count)
            with open(log, "rb") as f:
                size = f.seek(0, 2)
                start = j.get("offset", 0)
                if size < j.get("last_size", size) or size < start:
                    start = j["offset"] = 0  # truncated and rewritten since the last pass: read it all, from now on
                j["last_size"] = size
                f.seek(max(start, size - 200_000))
                tail = f.read().decode("utf-8", errors="replace")
        except Exception:
            continue
        try:
            done = bool(re.search(j.get("done") or r"(?!x)x", tail))
        except re.error as e:
            al.raise_(f"job:{name}", f"job {name}", f"invalid done regex {j.get('done')!r}: {e}")
            continue
        if done:
            event("DONE", f"job {name}", f"finished ({log.name})")
            finished.append(name)
            clear_job(al, name)
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
    jobs = load(JOBS, {})  # re-read so a job registered during this pass is kept
    for n in finished:
        jobs.pop(n, None)
    for n, j in jobs_now.items():
        if n in jobs and "last_size" in j and jobs[n].get("registered") == j.get("registered"):
            jobs[n]["last_size"], jobs[n]["offset"] = j["last_size"], j.get("offset", 0)
    save(JOBS, jobs)


def check_kaggle(st, al):
    ks, fails = st.setdefault("kernels", {}), st.setdefault("kaggle_fail", {})
    if not isinstance(fails, dict):
        fails = st["kaggle_fail"] = {}
    bad = ("ERROR", "CANCEL_REQUESTED", "CANCEL_ACKNOWLEDGED")
    for k in KERNELS:
        short = k.split("/")[1]
        code, out = run([KAGGLE, "kernels", "status", k], 60)
        m = re.search(r"KernelWorkerStatus\.(\w+)", out)
        if not m:  # count failures per kernel, keep checking the others
            fails[k] = fails.get(k, 0) + 1
            if fails[k] >= 5:
                al.raise_(f"kaggle-api:{short}", "kaggle", f"{short}: status calls failing: {out.strip()[-160:]}")
            continue
        fails[k] = 0
        al.clear(f"kaggle-api:{short}", "kaggle", f"{short}: API answering again")
        s, prev = m.group(1), ks.get(k)
        if s != prev:
            ks[k] = s
            if prev is None and s in bad:
                al.raise_(f"kernel:{short}", "kaggle", f"{short} is {s} (first seen)")
            elif prev is not None:
                lvl = "ALERT" if s in bad else ("DONE" if s == "COMPLETE" else "INFO")
                event(lvl, "kaggle", f"{short}: {prev} -> {s}")
            if s not in bad:
                al.clear(f"kernel:{short}")
    # An ERROR -> ERROR re-push is not visible here; after pushing a kernel the main agent re-checks it itself.


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
    first_pass = not st.get("subs_init")  # only the very first pass is silent (baseline), not a lost state file
    st["subs_init"] = True
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
    me = str(os.getpid())
    try:  # one pass at a time (Task Scheduler and a manual loop may overlap); a lock older than 10 min is stale
        try:
            if time.time() - LOCK.stat().st_mtime > 600:
                LOCK.unlink()
        except OSError:
            pass
        fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, me.encode())
        os.close(fd)
    except OSError:  # held by another pass (or delete-pending): skip this one
        return
    try:
        _pass()
    finally:
        try:
            if LOCK.read_text().strip() == me:  # never delete a newer pass's lock
                LOCK.unlink()
        except OSError:
            pass


def _pass():
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
    elif a[0] == "job" and len(a) >= 4:  # NAME LOG DONE_REGEX (required: how else would it know the job ended)
        log = Path(a[2]) if Path(a[2]).is_absolute() else ROOT / a[2]
        jobs = load(JOBS, {})
        jobs[a[1]] = {"log": a[2], "done": a[3], "stale_min": 30, "registered": time.time(),
                      "offset": log.stat().st_size if log.exists() else 0}
        save(JOBS, jobs)
        st = load(STATE, {})
        for k in (f"job:{a[1]}", f"jobstale:{a[1]}", f"jobmissing:{a[1]}"):
            st.get("alerts", {}).pop(k, None)
        save(STATE, st)
        print(f"watching job {a[1]}: {a[2]} (done when /{a[3]}/ appears)")
    elif a[0] == "unjob" and len(a) == 2:
        jobs = load(JOBS, {})
        jobs.pop(a[1], None)
        save(JOBS, jobs)
        st = load(STATE, {})
        for k in (f"job:{a[1]}", f"jobstale:{a[1]}", f"jobmissing:{a[1]}"):
            st.get("alerts", {}).pop(k, None)
        save(STATE, st)
    else:
        print(__doc__)
