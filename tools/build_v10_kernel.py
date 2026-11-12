"""Builds kernel_v10/ntl_v10.py: v9 (kernel_v9/ntl_v9.py) with the hostlike-trained Qwen member h3s2all replaced by
t4all (Qwen3-4B fine-tuned on train.csv only, no external stage 1; already in the ntl-qwen-weights dataset, used by
ntl-t4, LB 0.434 alone). Same weight 0.1, same post (centering 0.25, clusters N/9, partial sinkhorn 0.45).
Why submit although the simulator says -0.13 (research/sim_t4_for_h3.py): the user asked for a direct LB check, since
the simulator only resamples train.csv and cannot see how the hidden test differs. Clean member-swap probe vs v9 0.580.
Every replacement is asserted to match exactly once. Run: .venv/Scripts/python.exe tools/build_v10_kernel.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
src = (ROOT / "kernel_v9/ntl_v9.py").read_text(encoding="utf-8")
REPL = [
    ('"qwen:h3s2all": (0.1, 1.0)', '"qwen:t4all": (0.1, 1.0)'),
    ("# AUTO-BUILT by tools/build_v9_kernel.py from kernel_qwen/ntl_qwen_blend.py (v7 members, v9 post).\n",
     "# AUTO-BUILT by tools/build_v10_kernel.py from kernel_v9/ntl_v9.py: v9 with h3s2all replaced by t4all "
     "(train.csv-only Qwen).\n"),
]
out = src
for old, new in REPL:
    assert out.count(old) == 1, f"not found exactly once: {old[:70]!r}"
    out = out.replace(old, new)
w = out.split("WEIGHTS = {", 1)[1].split("}", 1)[0]
assert "h3s2all" not in w and '"qwen:t4all": (0.1, 1.0)' in w
assert (ROOT / "data/qwen_weights/t4all__weights.pt").exists()
compile(out, "ntl_v10.py", "exec")
d = ROOT / "kernel_v10"
d.mkdir(exist_ok=True)
(d / "ntl_v10.py").write_text(out, encoding="utf-8")
meta = json.load(open(ROOT / "kernel_v9/kernel-metadata.json"))
meta.update({"id": "varadarajborkar/ntl-v10", "title": "ntl-v10", "code_file": "ntl_v10.py"})
json.dump(meta, open(d / "kernel-metadata.json", "w"), indent=2)
print("wrote", d / "ntl_v10.py")
