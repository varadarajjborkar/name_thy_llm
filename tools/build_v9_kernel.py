"""Builds kernel_v9/ntl_v9.py: the shipped Qwen notebook (kernel_qwen/ntl_qwen_blend.py) with the v7 members
(the 8B member removed) and the post-processing chosen by the calibrated hidden-test simulator
(research/test_sim.py, posterior from the paired LB probes P1-P3; logs/test_sim_post.csv):
  prompt centering weight 0.5 -> 0.25, clusters round(N/18) -> round(N/9), partial sinkhorn 0.45 unchanged.
Simulated: +0.80 over the v7 post (SE 0.08), positive in every plausible cell (worst +0.45), which also makes this a
clean LB probe of post-processing alone (same members as v7, LB 0.571).
Every replacement is asserted to match exactly once. Run: .venv/Scripts/python.exe tools/build_v9_kernel.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
src = (ROOT / "kernel_qwen/ntl_qwen_blend.py").read_text(encoding="utf-8")
REPL = [
    ('           "qwen:k8all": (0.2, 1.0)}  # v8: Qwen3-8B stage 2 on top of v7: OOF +0.88, all 4 folds up '
     '(research/blend_8b.py)\n', None),
    ("CENTER_W = 0.5\n", "CENTER_W = 0.25  # v9: calibrated hidden-test simulator (research/test_sim.py)\n"),
    ("n_clusters=max(1, round(ok.sum() / len(classes)))", "n_clusters=max(1, round(ok.sum() / CLUSTER_SIZE))"),
]
out = src
for old, new in REPL:
    assert out.count(old) == 1, f"not found exactly once: {old[:70]!r}"
    if new is None:  # drop the 8B line: close the dict on the line before
        prev = '           "atlas": (0.15, 1.0),  # v7: on top of the v6 weights (they sum to 1), as evaluated out of fold\n'
        assert out.count(prev) == 1, "atlas weight line not found exactly once"
        out = out.replace(prev + old, '           "atlas": (0.15, 1.0)}  # v7 members (v9: 8B member dropped)\n')
    else:
        out = out.replace(old, new)
anchor = "CENTER_W = 0.25"
out = out.replace(anchor, "CLUSTER_SIZE = 9  # v9: about 4 passages per test prompt (simulator); clusters of ~9\n"
                  + anchor, 1)
out = "# AUTO-BUILT by tools/build_v9_kernel.py from kernel_qwen/ntl_qwen_blend.py (v7 members, v9 post).\n" + out
assert "k8all" not in out.split("WEIGHTS = {", 1)[1].split("}", 1)[0]
compile(out, "ntl_v9.py", "exec")
d = ROOT / "kernel_v9"
d.mkdir(exist_ok=True)
(d / "ntl_v9.py").write_text(out, encoding="utf-8")
meta = json.load(open(ROOT / "kernel_qwen/kernel-metadata.json"))
meta.update({"id": "varadarajborkar/ntl-v9", "title": "ntl-v9", "code_file": "ntl_v9.py",
             "model_sources": [m for m in meta["model_sources"] if "8b" not in m]})  # no 8B member any more
assert meta["model_sources"] == ["qwen-lm/qwen-3/transformers/4b-base/1"], meta["model_sources"]
json.dump(meta, open(d / "kernel-metadata.json", "w"), indent=2)
print("wrote", d / "ntl_v9.py")
