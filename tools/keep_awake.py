"""Keep Windows from sleeping for N hours (default 10), then exit. Nothing is changed permanently.

Run: .venv/Scripts/pythonw.exe tools/keep_awake.py 10
Closing the lid can still sleep the laptop if the lid action says so; this only blocks idle sleep.
"""
import ctypes
import sys
import time

ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
hours = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0
end = time.time() + hours * 3600
while time.time() < end:
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    time.sleep(60)
ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
