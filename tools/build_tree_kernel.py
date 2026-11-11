"""Builds kernel_tree/ntl_tree_kernel.py (one self-contained Kaggle script) from
features/style_features.py + tree/ntl_tree.py + tree/best_config.json + tree/kernel_main.py.
The `__main__` demo block of style_features.py and the import line containing `# BUILD: drop` are left out.
Run: .venv/Scripts/python.exe tools/build_tree_kernel.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sf = (ROOT / "features/style_features.py").read_text(encoding="utf-8")
assert sf.count('if __name__ == "__main__":') == 1
sf = sf.split('if __name__ == "__main__":')[0].rstrip() + "\n"
lib = (ROOT / "tree/ntl_tree.py").read_text(encoding="utf-8").splitlines()
drop = [ln for ln in lib if "# BUILD: drop" in ln]
assert len(drop) == 1, drop
lib = "\n".join(ln for ln in lib if "# BUILD: drop" not in ln) + "\n"
assert not [ln for ln in lib.splitlines() if ln.lstrip().startswith(("import style_features", "from style_features"))]
best = json.load(open(ROOT / "tree/best_config.json"))
config = {k: best[k] for k in ("design", "mode", "use_cg", "nodes", "flat", "cv") if k in best}
main = (ROOT / "tree/kernel_main.py").read_text(encoding="utf-8")
out = ROOT / "kernel_tree/ntl_tree_kernel.py"
out.parent.mkdir(exist_ok=True)
out.write_text(
    "# AUTO-BUILT by tools/build_tree_kernel.py: edit features/style_features.py, tree/ntl_tree.py,\n"
    "# tree/best_config.json or tree/kernel_main.py instead, then rebuild.\n"
    "# ======== features/style_features.py\n" + sf
    + "\n# ======== tree/ntl_tree.py\n" + lib
    + "\n# ======== tree/best_config.json\nCONFIG = " + json.dumps(config, indent=1) + "\n"
    + "\n" + main, encoding="utf-8")
compile(out.read_text(encoding="utf-8"), str(out), "exec")
meta = {"id": "varadarajborkar/ntl-tree", "title": "ntl-tree", "code_file": "ntl_tree_kernel.py",
        "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": False,
        "enable_internet": False, "dataset_sources": [], "competition_sources": ["name-that-llm"],
        "kernel_sources": [], "model_sources": []}
json.dump(meta, open(out.parent / "kernel-metadata.json", "w"), indent=2)
print("wrote", out, f"({len(out.read_text(encoding='utf-8').splitlines())} lines), design {config['design']},"
      f" mode {config['mode']}")
