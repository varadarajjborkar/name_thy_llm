"""Builds two Kaggle notebooks from the public notebook "Name That LLM | Deep Forensic EDA with Analogie" by
evgendvorkin (source pulled 2026-11-12, kept in external/ntl684_evgendvorkin/), whose Part II reproduces the public
0.684 "Clean LoRA + Qwen3-Embedding" ensemble. Purpose: measure that pipeline on our account and test one change.

  kernel_pub684/       ntl-pub684-copy : the original, code unchanged; only a credits cell is added at the top
  kernel_pub684_glm9/  ntl-pub684-glm9 : the same, with the Qwen3-8B-base LoRA24 reader "q8v5" (6 adapters, weight 2.0,
                       twin 0.35, Gaussian head 0.44) replaced by the GLM-4-9B LoRA24 reader "glm9v3" from
                       mathleteutd/ntl-two-stage-lora24-members-q8-glm9 (5 prompt-grouped fold fits + 1 full fit, author's
                       OOF 0.597), base lucasboesen/name-that-llm-glm4-9b-trunc25 (already attached to the original as a
                       kernel source). Same weight, twin and Gaussian settings; keep / layers / max_len / no_final_norm /
                       eos_id read from the adapter's meta.json at run time.
Every code replacement is asserted to match exactly once; the original's code cells are otherwise byte-identical.
Run: .venv/Scripts/python.exe tools/build_pub684.py
"""
import copy
import json
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "external/ntl684_evgendvorkin"
src_nb = nbf.read(SRC / "name-that-llm-deep-forensic-eda-with-analogie.ipynb", as_version=4)
src_meta = json.load(open(SRC / "kernel-metadata.json", encoding="utf-8"))
ORIG_URL = "https://www.kaggle.com/code/evgendvorkin/name-that-llm-deep-forensic-eda-with-analogie"

CREDIT_COMMON = f"""
> **This is a copy made only to measure the score on our own account. It is not our work.**
>
> Original notebook: **[Name That LLM | Deep Forensic EDA with Analogie]({ORIG_URL})** by **evgendvorkin** (pulled 2026-11-12).
> Its inference pipeline is the public **0.684 "Clean LoRA + Qwen3-Embedding" ensemble**; per its section 20 the work
> behind it belongs to **lbjyang** (NTL 684 Ozertuu Mirror), **Vladislav Lavrentev** (lightningv08), **CHEN Han**
> (concyclics), **Omer Faruk Merey**, **Omkar Pattanaik**, **Areeb Ali**, **Dhruv Tibarewal**, **Nishant Dahal**,
> **Joseph Adamski**, **aboodps**, **heliosli**, **Roman Tamrazov**, **mercurydust**, **prvsiyan**, **kirilltym**,
> **Lucas Boesen**, **sangam21**, **pjmathematician**, **vnton37nguyn**, **zbigniewkirchner**, **crtproduction** and the
> other authors of the attached datasets and models. All fitted weights, adapters and the inference architecture are theirs.
"""
CREDIT_COPY = "# Copy for evaluation (unchanged)\n" + CREDIT_COMMON + """
> **Changes in this copy:** none to the code. Only this cell was added.
"""
CREDIT_GLM = "# Variant for evaluation: GLM-4-9B reader instead of Qwen3-8B\n" + CREDIT_COMMON + """
> **One change in this variant** (cells 15 and 41 of the original): the Qwen3-8B-base LoRA24 reader `q8v5` is replaced by
> the GLM-4-9B LoRA24 reader `glm9v3` with the same weight and heads. GLM adapters by **mathleteutd**
> ([ntl-two-stage-lora24-members-q8-glm9](https://www.kaggle.com/datasets/mathleteutd/ntl-two-stage-lora24-members-q8-glm9),
> recipe after Nishant Dahal and Dhruv Tibarewal); GLM-4-9B truncated base by **Lucas Boesen**
> (name-that-llm-glm4-9b-trunc25). Two safety checks are added to cell 41: a GLM smoke test on the public test rows in the
> quick path, and a stop if the GLM reader fails during the full run. Everything else is unchanged.
"""

Q8_SPEC = ('    {"name": "q8v5", "type": "l24", "path": B8, "adapters": AQ8V5, "keep": 24, "layers": LAYERS, "w": 2.0, '
           '"bs": 16, "max_len": 384, "twin": 0.35, "gauss": 0.44, "dp": True},')
GLM_SPEC = ('    {"name": "glm9v3", "type": "l24", "path": BGLM, "adapters": AGLM9, "keep": GLM9C["keep"], '
            '"layers": GLM9C["layers"], "w": 2.0, "bs": 16, "max_len": GLM9C["max_len"], "twin": 0.35, "gauss": 0.44, '
            '"dp": True, "no_final_norm": GLM9C["no_final_norm"], "eos_id": GLM9C["eos_id"]},')
Q8_SETUP = "AQ8V5 = [adapter_dir(f'q8v5_f{k}') for k in range(5)] + [adapter_dir('q8v5')]"
GLM_SETUP = (Q8_SETUP + "\n"
             "# variant: GLM-4-9B LoRA24 reader (mathleteutd/ntl-two-stage-lora24-members-q8-glm9) on Lucas Boesen's\n"
             "# truncated GLM-4-9B base; settings from the adapter's own meta.json\n"
             "AGLM9 = [adapter_dir(f'glm9v3_f{k}') for k in range(5)] + [adapter_dir('glm9v3')]\n"
             "_, GLM9C = meta_dir('glm9v3')\n"
             "BGLM = model_dir('glm4_9b_trunc25')\n"
             "assert GLM9C['layers'][-1] <= GLM9C['keep'] and len(GLM9C['layers']) == 8, GLM9C\n"
             "print('GLM reader', BGLM, GLM9C)")


RUN_OLD = ("    print('prefit core votes smoke ok', {k: v.shape for k, v in smoke.items()})\n"
           "else:\n"
           "    print('infer exit code', run(cmd, env=dict(os.environ, NTL_PIPE_WORKERS='1')))")
# issue 123: (1) at push time (FAST_COMMIT) really load the GLM reader once on the public test rows, with the same steps
# as infer.py's l24_predict (truncate, no final norm, merge the LoRA, masked mean at the 8 depths, LayerNorm, plain
# head); (2) at scoring, fail loudly if the GLM member failed or did not finish all 6 adapters, instead of silently
# scoring a pipeline without it.
RUN_NEW = r"""    print('prefit core votes smoke ok', {k: v.shape for k, v in smoke.items()})
    # variant smoke test: load the GLM reader on the public test rows, same steps as infer.py's l24_predict
    import gc
    import torch
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    from safetensors.torch import load_file
    tok = AutoTokenizer.from_pretrained(BGLM)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    gcfg = AutoConfig.from_pretrained(BGLM); gcfg.num_hidden_layers = GLM9C['keep']; gcfg.use_cache = False
    gm = AutoModel.from_pretrained(BGLM, config=gcfg, dtype=torch.float16, device_map='auto').eval()
    gm.requires_grad_(False)  # in-place LoRA merge below (infer.py's l24_predict runs under @torch.no_grad())
    if GLM9C['no_final_norm']: gm.norm = torch.nn.Identity()
    ad = AGLM9[-1]
    acfg = json.load(open(os.path.join(ad, 'adapter_config.json'))); scale = acfg['lora_alpha'] / acfg['r']
    sd = load_file(os.path.join(ad, 'adapter_model.safetensors')); mods = dict(gm.named_modules()); n_merged = 0
    with torch.no_grad():
        for k, v in sd.items():
            if '.lora_A.' not in k: continue
            name = k.replace('base_model.model.', '', 1).split('.lora_A.')[0]
            if name not in mods and ('model.' + name) in mods: name = 'model.' + name
            W = mods[name].weight; B = sd[k.replace('.lora_A.', '.lora_B.')]
            W += ((B.to(W.device, torch.float32) @ v.to(W.device, torch.float32)) * scale).to(W.dtype); n_merged += 1
    del mods, W, B, v
    H = torch.load(os.path.join(ad, 'heads.pt'), map_location='cpu')
    texts = pd.read_csv(os.path.join(DATA, 'test.csv')).text.fillna('').tolist()
    enc = tok(texts, truncation=True, max_length=GLM9C['max_len'])['input_ids']
    width = max(len(x) for x in enc)
    ids = torch.full((len(enc), width), tok.pad_token_id); att = torch.zeros((len(enc), width), dtype=torch.long)
    for r, x in enumerate(enc): ids[r, :len(x)] = torch.tensor(x); att[r, :len(x)] = 1
    dev = next(gm.parameters()).device
    with torch.no_grad():
        hs = gm(input_ids=ids.to(dev), attention_mask=att.to(dev), output_hidden_states=True, use_cache=False).hidden_states
        if not all(torch.isfinite(hs[l]).all() for l in GLM9C['layers']):  # fp16 overflow: redo in bf16, as infer.py does
            print('GLM smoke: fp16 overflow, redoing in bf16')
            gm.to(torch.bfloat16)
            hs = gm(input_ids=ids.to(dev), attention_mask=att.to(dev), output_hidden_states=True, use_cache=False).hidden_states
    mask = att.unsqueeze(-1).float(); mask[:, 0] = 0
    feats = []
    for i, l in enumerate(GLM9C['layers']):
        f = (hs[l].float().cpu() * mask).sum(1) / mask.sum(1)
        ln = torch.nn.LayerNorm(f.shape[1]); ln.load_state_dict({'weight': H[f'norms.{i}.weight'], 'bias': H[f'norms.{i}.bias']})
        feats.append(ln(f))
    with torch.no_grad():
        logits = torch.cat(feats, -1) @ H['plain.weight'].float().T + H['plain.bias'].float()
    glabels = json.load(open(os.path.join(ad, 'meta.json')))['labels']
    assert n_merged == 6 * GLM9C['keep'] and torch.isfinite(logits).all(), (n_merged, logits)
    print('GLM smoke ok:', n_merged, 'LoRA pairs merged; public-test predictions', [glabels[i] for i in logits.argmax(1).tolist()])
    del gm, sd, hs, feats, logits; gc.collect(); torch.cuda.empty_cache()
else:
    INFER_LOG = []
    def run_tee(cmd, env=None):
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env, bufsize=1)
        for line in proc.stdout:
            INFER_LOG.append(line)
            if not re.search(r'Warning|warn', line):
                print(line, end='')
        return proc.wait()
    print('infer exit code', run_tee(cmd, env=dict(os.environ, NTL_PIPE_WORKERS='1')))
    # variant guard: the GLM reader must really be in the blend; otherwise stop instead of scoring a pipeline without it
    glm_done = sum(('l24 glm9v3: adapter' in l) and (' done' in l) for l in INFER_LOG)
    glm_failed = any('MEMBER FAILED glm9v3' in l or 'MEMBER SKIPPED (time): glm9v3' in l for l in INFER_LOG)
    assert glm_done == 6 and not glm_failed, f'GLM reader incomplete: {glm_done} of 6 adapters done, failed={glm_failed}'
    print('GLM reader complete:', glm_done, 'adapters')"""


def code_cells(nb):
    return [c for c in nb.cells if c.cell_type == "code"]


def replace_once(nb, old, new):
    hits = [c for c in code_cells(nb) if old in c.source]
    assert len(hits) == 1 and hits[0].source.count(old) == 1, f"not found exactly once: {old[:60]!r}"
    hits[0].source = hits[0].source.replace(old, new)


def build(nb, credit, kid, title, extra_sources, outdir, fname):
    nb = copy.deepcopy(nb)
    for c in code_cells(nb):  # no stale outputs from the author's run
        c.outputs, c.execution_count = [], None
    nb.cells.insert(0, nbf.v4.new_markdown_cell(credit.strip()))
    meta = dict(src_meta)
    for k in ("id_no",):
        meta.pop(k, None)
    meta.update({"id": kid, "title": title, "code_file": fname, "is_private": True,
                 "dataset_sources": src_meta["dataset_sources"] + extra_sources})
    d = ROOT / outdir
    d.mkdir(exist_ok=True)
    nbf.write(nb, d / fname)
    json.dump(meta, open(d / "kernel-metadata.json", "w", encoding="utf-8"), indent=2)
    return nb


orig = build(src_nb, CREDIT_COPY, "varadarajborkar/ntl-pub684-copy", "ntl-pub684-copy", [],
             "kernel_pub684", "ntl-pub684-copy.ipynb")
var = copy.deepcopy(src_nb)
replace_once(var, Q8_SPEC, GLM_SPEC)
replace_once(var, Q8_SETUP, GLM_SETUP)
replace_once(var, RUN_OLD, RUN_NEW)
var = build(var, CREDIT_GLM, "varadarajborkar/ntl-pub684-glm9", "ntl-pub684-glm9",
            ["mathleteutd/ntl-two-stage-lora24-members-q8-glm9"], "kernel_pub684_glm9", "ntl-pub684-glm9.ipynb")

# checks: the copy's code is identical to the source; the variant differs only in the two replaced spots
for a, b in zip(code_cells(src_nb), code_cells(orig)):
    assert a.source == b.source
diff = [i for i, (a, b) in enumerate(zip(code_cells(src_nb), code_cells(var))) if a.source != b.source]
assert len(diff) == 2, diff
assert '"name": "q8v5"' not in "".join(c.source for c in code_cells(var))
print("wrote kernel_pub684/ and kernel_pub684_glm9/; variant differs in code cells", diff)
