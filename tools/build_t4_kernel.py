"""Builds kernel_t4/ntl_t4.py: the shipped Qwen notebook (kernel_qwen/ntl_qwen_blend.py) with ONE member,
qwen:t4all (Qwen3-4B fine-tuned on train.csv only, no external data), plus the notebook's usual post-processing
(prompt centering 0.5 + partial sinkhorn 0.45). Everything else in the notebook is untouched, so this is the same
tested code path with a different WEIGHTS dict. OOF of the t4 family: raw 0.398, after post 0.490.
Run: .venv/Scripts/python.exe tools/build_t4_kernel.py
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
src = (ROOT / "kernel_qwen/ntl_qwen_blend.py").read_text(encoding="utf-8")
pat = re.compile(r"^WEIGHTS = \{.*?\}  # v8:[^\n]*\n", re.S | re.M)
assert len(pat.findall(src)) == 1, "WEIGHTS block not found exactly once"
new = ('WEIGHTS = {"qwen:t4all": (1.0, 1.0)}  # t4 experiment: Qwen3-4B on train.csv only '
       '(OOF raw 0.398, post 0.490)\n')
out_src = ("# AUTO-BUILT by tools/build_t4_kernel.py from kernel_qwen/ntl_qwen_blend.py (only WEIGHTS differs).\n"
           + pat.sub(new, src))
compile(out_src, "ntl_t4.py", "exec")
d = ROOT / "kernel_t4"
d.mkdir(exist_ok=True)
(d / "ntl_t4.py").write_text(out_src, encoding="utf-8")
meta = json.load(open(ROOT / "kernel_qwen/kernel-metadata.json"))
meta.update({"id": "varadarajborkar/ntl-t4", "title": "ntl-t4", "code_file": "ntl_t4.py",
             "dataset_sources": ["varadarajborkar/ntl-qwen-weights", "varadarajborkar/ntl-wheels"],
             "model_sources": ["qwen-lm/qwen-3/transformers/4b-base/1"]})
json.dump(meta, open(d / "kernel-metadata.json", "w"), indent=2)
print("wrote", d / "ntl_t4.py", "| WEIGHTS:", new.strip())
