"""Control the resumable training run.

  .venv\\Scripts\\python.exe finetune\ctl.py start [train.py args]   start, or resume after a pause/crash/power cut
  .venv\\Scripts\\python.exe finetune\ctl.py pause                   save at the next step boundary, then stop
  .venv\\Scripts\\python.exe finetune\ctl.py status                  running? step, loss, accuracy, ETA
  .venv\\Scripts\\python.exe finetune\ctl.py tail [N]                last N log lines
  .venv\\Scripts\\python.exe finetune\ctl.py watchdog-install   auto-restart if it dies or hangs, also after a reboot
  .venv\\Scripts\\python.exe finetune\ctl.py watchdog-remove    remove that scheduled task
  .venv\\Scripts\\python.exe finetune\ctl.py kill                    hard stop (loses work since the last checkpoint)

Add  --run NAME  before the command to use another run (default run1).
The supervisor restarts a crashed trainer from the last checkpoint (up to 5 failures in a row without progress).
Training itself needs no internet.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
argv = sys.argv[1:]
run = "run1"
if argv[:1] == ["--run"]:
    run, argv = argv[1], argv[2:]
ACTIVE = ROOT / "checkpoints" / "ACTIVE_RUN"  # the run most recently started; the watchdog follows it
if argv[:1] == ["watchdog"] and ACTIVE.exists() and ACTIVE.read_text().strip():
    run = ACTIVE.read_text().strip()
RUN = ROOT / "checkpoints" / run
RUN.mkdir(parents=True, exist_ok=True)
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
FATAL = {3, 4}  # non-finite loss, config mismatch: retrying cannot help


def alive(pid):
    """True only if `pid` is really our supervisor (a stale PID after a reboot may belong to another program)."""
    if not pid:
        return False
    ps = f"(Get-CimInstance Win32_Process -Filter 'ProcessId={int(pid)}').CommandLine"
    out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True).stdout
    return "_supervise" in out and "ctl.py" in out


def sup_pid():
    f = RUN / "supervisor.pid"
    return int(f.read_text()) if f.exists() and f.read_text().strip().isdigit() else 0


def progress():
    f = RUN / "progress.json"
    try:
        return json.loads(f.read_text())
    except Exception:
        return {"step": 0}


def slog(msg):
    with open(RUN / "supervisor.log", "a", encoding="utf-8") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + msg + "\n")


def supervise(train_args):
    (RUN / "supervisor.pid").write_text(str(os.getpid()))
    fails = 0
    while True:
        if (RUN / "DONE_TRAINING").exists():
            slog("training finished")
            break
        if (RUN / "PAUSE").exists() or (RUN / "STOP").exists():
            slog("PAUSE/STOP file present, not starting")
            break
        before = progress().get("step", 0)
        slog(f"launching trainer (resume from step {before})")
        env = dict(os.environ, PYTHONUNBUFFERED="1", TRANSFORMERS_VERBOSITY="error", HF_HUB_DISABLE_PROGRESS_BARS="1",
                   TOKENIZERS_PARALLELISM="false")
        with open(RUN / "console.log", "a", encoding="utf-8") as con:
            rc = subprocess.call([PY, str(ROOT / "finetune" / "train.py"), "--run", run] + train_args, stdout=con,
                                 stderr=subprocess.STDOUT, env=env, cwd=ROOT)
        slog(f"trainer exited with code {rc}")
        if rc == 0:
            break
        if rc == 10:
            slog("paused")
            break
        if rc in FATAL:
            slog("fatal error, not retrying (see train.log)")
            (RUN / "HALTED").write_text(f"fatal exit code {rc}, see train.log")
            break
        after = progress().get("step", 0)
        fails = 0 if after > before else fails + 1
        if fails >= 5:
            slog("5 failures in a row without progress, giving up (see console.log)")
            (RUN / "HALTED").write_text("5 crashes in a row without progress, see console.log")
            break
        wait = min(30 * 2 ** fails, 600)
        slog(f"crash; retrying in {wait}s (failures without progress: {fails})")
        time.sleep(wait)
    (RUN / "supervisor.pid").write_text("0")


def launch_detached():
    """Start the supervisor through WMI so it lives OUTSIDE the caller's Windows job object.

    A plain DETACHED_PROCESS child still dies when the terminal that launched it is closed (terminals put their
    children in a kill-on-close job). WMI-created processes are not part of that job.
    """
    cmd = f'"{PY}" "{Path(__file__).resolve()}" --run {run} _supervise'.replace("'", "''")
    # ShowWindow 0: no console window (on Windows 11 each one opens a Windows Terminal window, and closing it kills the run)
    ps = ("$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}; "
          "$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments "
          f"@{{CommandLine='{cmd}'; CurrentDirectory='{ROOT}'; ProcessStartupInformation=$si}}; "
          "if ($r.ReturnValue -ne 0) { exit 1 }; $r.ProcessId")
    out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True)
    if out.returncode != 0 or not out.stdout.strip().split():
        raise SystemExit(f"could not start the supervisor: {out.stderr.strip() or out.stdout.strip()}")
    return int(out.stdout.strip().split()[-1])


def kill_trainer():
    ps = ("Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { $_.ProcessId -ne $PID -and "
          "$_.CommandLine -match 'train\.py.* --run " + run + "(\s|$)' } | "
          "ForEach-Object { Stop-Process -Id $_.ProcessId -Force }")
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True)


def wlog(msg):
    with open(RUN / "watchdog.log", "a", encoding="utf-8") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + msg + "\n")


HANG_SECONDS = 30 * 60  # train.log normally updates every ~5 min; the longest quiet stretch is a ~7 min full eval


def watchdog():
    """Run every few minutes by Task Scheduler. Revives a dead supervisor and unsticks a hung trainer."""
    if any((RUN / f).exists() for f in ("DONE_TRAINING", "PAUSE", "STOP", "HALTED")):
        return  # finished, or the user (or a fatal error) asked for it to stay down
    if not alive(sup_pid()):
        wlog(f"supervisor not running, restarting (checkpoint step {progress().get('step', 0)})")
        (RUN / "supervisor.pid").write_text(str(launch_detached()))
        return
    lf = RUN / "train.log"
    if lf.exists() and time.time() - lf.stat().st_mtime > HANG_SECONDS:
        wlog("train.log silent for 30+ minutes, killing the trainer so the supervisor restarts it from the last checkpoint")
        kill_trainer()
        lf.touch()  # restart the clock so we do not kill it again while it reloads


TASK = "NameThatLLM-Watchdog"


def watchdog_install():
    pyw = str(ROOT / ".venv" / "Scripts" / "pythonw.exe")
    ctl = str(Path(__file__).resolve())
    ps = f"""
$a = New-ScheduledTaskAction -Execute '{pyw}' -Argument '"{ctl}" --run {run} watchdog' -WorkingDirectory '{ROOT}'
$t1 = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) -RepetitionInterval (New-TimeSpan -Minutes 5) -RepetitionDuration (New-TimeSpan -Days 3650)
$t2 = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$s = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 10)
Register-ScheduledTask -TaskName '{TASK}' -Action $a -Trigger @($t1, $t2) -Settings $s -Force | Out-Null
(Get-ScheduledTask -TaskName '{TASK}').State
"""
    out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True)
    print(f"watchdog task '{TASK}':", out.stdout.strip() or out.stderr.strip())


def main():
    cmd = argv[0] if argv else "status"
    rest = argv[1:]
    if cmd == "_supervise":
        args_f = RUN / "args.json"
        supervise(json.loads(args_f.read_text()) if args_f.exists() else [])
    elif cmd == "start":
        if alive(sup_pid()) and (RUN / "PAUSE").exists():
            print("the run is still saving its pause checkpoint, waiting for it to exit...")
            for _ in range(60):
                time.sleep(3)
                if not alive(sup_pid()):
                    break
        if alive(sup_pid()):
            print("already running. Use: ctl.py status")
            return
        if (RUN / "DONE_TRAINING").exists():
            print("this run already finished. Use a new --run name.")
            return
        for f in ("PAUSE", "STOP", "HALTED"):
            (RUN / f).unlink(missing_ok=True)
        args_f = RUN / "args.json"
        has_ckpt = any((RUN / d / "OK").exists() for d in ("ckpt_latest", "ckpt_prev", "ckpt_tmp"))
        if rest and (not args_f.exists() or not has_ckpt):
            args_f.write_text(json.dumps(rest))  # nothing to stay consistent with yet, so new args may replace old ones
        elif rest:
            print(f"note: run already has saved settings {json.loads(args_f.read_text())}; ignoring the new ones.")
        pid = launch_detached()
        (RUN / "supervisor.pid").write_text(str(pid))
        ACTIVE.write_text(run)
        print(f"started run '{run}'. Check with: ctl.py status")
    elif cmd == "pause":
        (RUN / "PAUSE").write_text("pause")
        print("pause requested. The trainer saves at the end of the current step (under a minute, up to a few"
              " minutes during an eval) and exits. Resume with: ctl.py start")
    elif cmd == "status":
        pid = sup_pid()
        pr = progress()
        print(f"run {run}: supervisor {'RUNNING' if alive(pid) else 'not running'}"
              f"{' (pause requested)' if (RUN / 'PAUSE').exists() else ''}"
              f"{', FINISHED' if (RUN / 'DONE_TRAINING').exists() else ''}"
              f"{', STOPPED by user' if (RUN / 'STOP').exists() else ''}")
        if (RUN / "HALTED").exists():
            print("HALTED:", (RUN / "HALTED").read_text())
        print(f"last checkpoint: step {pr.get('step', 0)}/{pr.get('total', '?')}  best subset acc {pr.get('best', 0):.4f}")
        lf = RUN / "train.log"
        if lf.exists():
            print("--- train.log")
            print("".join(lf.read_text(encoding="utf-8", errors="replace").splitlines(True)[-4:]))
        sl = RUN / "supervisor.log"
        if sl.exists():
            print("--- supervisor.log")
            print("".join(sl.read_text(encoding="utf-8").splitlines(True)[-3:]))
    elif cmd == "tail":
        n = int(rest[0]) if rest else 20
        lf = RUN / "train.log"
        print("".join(lf.read_text(encoding="utf-8", errors="replace").splitlines(True)[-n:]) if lf.exists() else "no log yet")
    elif cmd == "kill":
        (RUN / "STOP").write_text("stopped by user")  # tells the watchdog not to bring it back
        pid = sup_pid()
        if pid:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], capture_output=True)
        kill_trainer()
        (RUN / "supervisor.pid").write_text("0")
        print("killed (work since the last checkpoint is lost). Resume with: ctl.py start")
    elif cmd == "watchdog":
        watchdog()
    elif cmd == "watchdog-install":
        watchdog_install()
    elif cmd == "watchdog-remove":
        subprocess.run(["schtasks", "/delete", "/tn", TASK, "/f"])
    else:
        print(__doc__)


main()
