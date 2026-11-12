"""Builds research/data_compare.ipynb: Kaggle train.csv against the external data our models learned from,
side by side, mostly as heatmaps. Datasets: Kaggle (2,699), Pool 92k (pool_cont2: stage 1 of the main model s4),
Pool 60k (pool_typo_60k: stage 1 of the older Qwen members), Hostlike (prvsiyan, 6,897).
Feature charts use class-matched samples (up to 150 texts per model from every set, like Kaggle), so a difference
is never just a different model mix. Style features are features/style_features.py (the LightGBM member's features).
Run: .venv/Scripts/python.exe research/make_data_compare.py
then: cd research && ../.venv/Scripts/python.exe -m jupyter nbconvert --to notebook --execute --inplace data_compare.ipynb
The "what this shows" texts were written after reading the executed outputs; re-check them if the data changes.
"""
from pathlib import Path

import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip()))  # noqa: E731
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip()))  # noqa: E731

md("""
# Kaggle data vs external data, side by side

Our models first learn from **external data** (texts other people generated from the same 18 models), then get a short polish on Kaggle's `train.csv`. This notebook puts the datasets next to each other so you can see where they agree and where they drift.

| Name here | File | Rows | Used for |
|---|---|---|---|
| **Kaggle** | `data/train.csv` | 2,699 | stage 2 (the polish), all scoring |
| **Pool 92k** | `pool_cont2` | 92,485 | stage 1 of the main model (s4, weight 0.5 in v9) |
| **Pool 60k** | `pool_typo_60k` | 60,000 | stage 1 of the older Qwen members |
| **Hostlike** | `hostlike` | 6,897 | added to stage 1 for one older member |

**How to read the colours**
- **Amounts** (shares, lengths, similarities) use one colour going from light to dark: darker means more.
- **Differences** (external minus Kaggle) use blue for "less than Kaggle", grey for "the same", orange for "more than Kaggle". Grey everywhere would mean a perfect match.
- Model names are coloured by family (Claude blue, Gemini orange, GPT green, GLM yellow, Kimi pink, Qwen dark green).

**Fair comparison:** for every feature chart, each external set is cut down to at most 150 texts per model, the same as Kaggle. So a difference never comes from one set simply having more GPT texts.
""")

code(r'''
import sys, warnings
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
warnings.filterwarnings("ignore")

ROOT = "../"
sys.path.insert(0, ROOT + "features")
import style_features as SF

FILES = {"Kaggle": "data/train.csv", "Pool 92k": "data/processed/pool_cont2.parquet",
         "Pool 60k": "data/processed/pool_typo_60k.parquet", "Hostlike": "data/processed/hostlike.parquet"}
import pyarrow.parquet as pq
def load(f):
    if f.endswith(".csv"):
        d = pd.read_csv(ROOT + f); d["grp"] = "kaggle:" + d.prompt_id.astype(str); return d
    has = "prompt_group" in pq.ParquetFile(ROOT + f).schema_arrow.names
    d = pd.read_parquet(ROOT + f, columns=["text", "model"] + (["prompt_group"] if has else []))
    # prompt ids (for splits that keep one prompt on one side); Pool 92k has none, so each text is its own group
    d["grp"] = (f + ":" + d.prompt_group.astype(str)) if has else (f + ":row" + pd.Series(range(len(d)), index=d.index).astype(str))
    return d
FULL = {k: load(f) for k, f in FILES.items()}
for d in FULL.values(): d["text"] = d.text.fillna("").astype(str)
TX = lambda X: np.sign(X) * np.log1p(np.abs(X))  # signed log: keeps sentinels like -1 and negative readability
SETS = list(FULL)
EXT = SETS[1:]

FAMILIES = ["claude", "gemini", "gpt", "glm", "kimi", "qwen"]
FAM_COLOR = dict(zip(FAMILIES, ["#2a78d6", "#eb6834", "#1baf7a", "#c08400", "#e87ba4", "#008300"]))
fam = lambda m: next(f for f in FAMILIES if m.startswith(f))
MODELS = sorted(FULL["Kaggle"].model.unique(), key=lambda m: (FAMILIES.index(fam(m)), m))
for k in EXT: FULL[k] = FULL[k][FULL[k].model.isin(MODELS)]

SEQ = LinearSegmentedColormap.from_list("seq", ["#f5f8fc", "#b9d3f2", "#5b93db", "#1c5cab", "#0b2f5e"])
DIV = LinearSegmentedColormap.from_list("div", ["#1c5cab", "#86b6ef", "#ecebe6", "#f3a17f", "#b8431a"])
plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})

def colour_labels(ax, axis="y"):
    for lab in (ax.get_yticklabels() if axis == "y" else ax.get_xticklabels()):
        t = lab.get_text()
        if any(t.startswith(f) for f in FAMILIES):
            lab.set_color(FAM_COLOR[fam(t)]); lab.set_fontweight("bold")

def heat(ax, M, rows, cols, cmap, norm=None, vmin=None, vmax=None, fmt="{:.2f}", annot=True, cbar=True, label=""):
    """Heatmap with numbers in the cells; text turns white on dark cells so it stays readable."""
    M = np.asarray(M, dtype=float)
    im = ax.imshow(M, cmap=cmap, norm=norm, vmin=None if norm else vmin, vmax=None if norm else vmax, aspect="auto")
    ax.set_xticks(range(len(cols)), cols, rotation=45, ha="right"); ax.set_yticks(range(len(rows)), rows)
    ax.set_xticks(np.arange(-.5, len(cols)), minor=True); ax.set_yticks(np.arange(-.5, len(rows)), minor=True)
    ax.grid(which="minor", color="white", lw=1.5); ax.tick_params(which="minor", length=0)
    for s in ax.spines.values(): s.set_visible(False)
    if annot:
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if np.isfinite(M[i, j]):
                    r, g, b, _ = im.cmap(im.norm(M[i, j]))
                    ax.text(j, i, fmt.format(M[i, j]), ha="center", va="center", fontsize=8,
                            color="white" if 0.299 * r + 0.587 * g + 0.114 * b < 0.5 else "#222")
    if cbar: plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label=label)
    colour_labels(ax, "y"); colour_labels(ax, "x")
    return im

# class-matched samples: at most 150 texts per model from every set (Kaggle has 150 per model)
S = {"Kaggle": FULL["Kaggle"][["text", "model", "grp"]].copy()}
for k in EXT:
    d = FULL[k]
    S[k] = pd.concat([d[d.model == m].sample(min(150, (d.model == m).sum()), random_state=0) for m in MODELS])
NAMES = list(SF.NAMES)
F = {k: pd.DataFrame([SF.extract(t) for t in s.text], columns=NAMES, index=s.index) for k, s in S.items()}
VARY = [n for n in NAMES if F["Kaggle"][n].std() > 0]
for k in SETS: print(f"{k:9s} {len(FULL[k]):6d} texts in total, {len(S[k])} in the class-matched sample")
print(f"{len(NAMES)} style features, {len(VARY)} vary inside Kaggle")
''')

# ---------------------------------------------------------------- 1. model mix
md("""
## 1. How many texts per model?

Each cell is the share of that dataset's texts written by that model. A perfectly balanced set has 5.6% everywhere (1 of 18): grey. Blue = fewer texts than a fair share, orange = more.
""")
code(r'''
share = pd.DataFrame({k: FULL[k].model.value_counts(normalize=True).reindex(MODELS).fillna(0) * 100 for k in SETS})
fig, ax = plt.subplots(figsize=(6.5, 8))
heat(ax, share.values, MODELS, SETS, DIV, norm=TwoSlopeNorm(100 / 18, 2, 9.1), fmt="{:.1f}",
     label="% of the dataset (grey = fair share 5.6)")
ax.set_title("Share of texts per model (%)", loc="left"); plt.tight_layout(); plt.show()
''')
md("""
**What this shows:**
- **Pool 60k** was sampled to be balanced: grey everywhere, like Kaggle.
- **Pool 92k is uneven.** claude-opus-5 has only 2.6%, gemini-3.7-flash and qwen3.8-max 3.2%, kimi-k3 4.6%, while most others have 6.5%.
- **Hostlike** has extra GPT texts (6.3% to 7.2%) and few Geminis (gemini-3.7-flash 3.5%, gemini-3.8-flash 4.2%).
- So **gemini-3.7-flash is short of texts in both sets that fed the main model's training**, and it is one of the models we recognise worst. The main model saw fewer examples of it during its first stage.
""")

# ---------------------------------------------------------------- 2. length
md("""
## 2. How long are the texts?

Left: the spread of lengths in each dataset (the middle 50% is the box, the line is the median).
Right: the median length per model, divided by Kaggle's median for the same model. **1.00 (grey) means the same length as Kaggle**, blue means shorter, orange means longer.
""")
code(r'''
words = {k: S[k].text.str.split().str.len() for k in SETS}
fig, axes = plt.subplots(1, 2, figsize=(14, 8), gridspec_kw={"width_ratios": [1, 1.1]})
ax = axes[0]
DS_COLOR = {"Kaggle": "#222222", "Pool 92k": "#6250d6", "Pool 60k": "#a99ff0", "Hostlike": "#e34948"}
bp = ax.boxplot([words[k] for k in SETS], vert=True, widths=0.55, patch_artist=True, showfliers=False,
                medianprops={"color": "white", "lw": 2})
for p, k in zip(bp["boxes"], SETS): p.set_facecolor(DS_COLOR[k]); p.set_edgecolor(DS_COLOR[k])
ax.set_xticks(range(1, 5), SETS); ax.set_ylabel("words per text"); ax.grid(axis="y", color="#eee"); ax.set_axisbelow(True)
for i, k in enumerate(SETS, 1): ax.text(i, np.median(words[k]) + 3, f"{np.median(words[k]):.0f}", ha="center", fontsize=9)
ax.set_title("Length of texts (class-matched)", loc="left")
med = pd.DataFrame({k: words[k].groupby(S[k].model).median().reindex(MODELS) for k in SETS})
ratio = med[EXT].div(med["Kaggle"], axis=0)
heat(axes[1], ratio.values, MODELS, EXT, DIV, norm=TwoSlopeNorm(1.0, 0.65, 1.35), fmt="{:.2f}",
     label="median length vs Kaggle (1 = same)")
axes[1].set_title("Median length per model, relative to Kaggle", loc="left")
plt.tight_layout(); plt.show()
print("Kaggle median words per model:", med["Kaggle"].astype(int).to_dict())
''')
md("""
**What this shows:**
- Overall, the two pools have Kaggle's length (median 101 and 100 words vs 101). **Hostlike is shorter** (86): blue in almost every row of the right panel, down to 0.69 for gemini-3.7-flash.
- Per model, the pools mostly sit within 10% of Kaggle. The exceptions: kimi-k3 is longer outside (1.15 and 1.11), and in Pool 60k glm-5.3 (1.18) and claude-sonnet-5 (1.15).
- **A clue that gets lost:** in Kaggle, gemini-3.7-flash writes clearly longer texts than gemini-3.8-flash (median 101 vs 84 words). In Pool 92k the gap shrinks to about 94 vs 87, and in hostlike the two are about equal (70 vs 77; with 150 texts per model a median can move by several words, so the order there is within noise). So the external data teaches "longer means 3.7" only weakly, and 3.7 vs 3.8 is one of our worst mix-ups.
""")

# ---------------------------------------------------------------- 3. habits
md("""
## 3. Writing habits, model by model

Six habits from our style features. In each small heatmap, a row is a model and a column is a dataset, and the colour is the raw value (darker = more of that habit). If the external data copies Kaggle, **each row has one even colour from left to right**.
""")
code(r'''
HABITS = {"emdash per 100 chars": "p_emdash", "curly quotes per 100 chars": "p_rdquo",
          "semicolons per 100 chars": "p_semicolon", "colons per 100 chars": "p_colon",
          "words per sentence": None, "mean word length": "mean_word_len"}
sent_col = next((n for n in NAMES if "sent" in n and "mean" in n), None)
HABITS["words per sentence"] = sent_col
HABITS = {k: v for k, v in HABITS.items() if v in NAMES}
fig, axes = plt.subplots(2, 3, figsize=(16, 15))
for ax, (title, col) in zip(axes.ravel(), HABITS.items()):
    M = pd.DataFrame({k: F[k][col].groupby(S[k].model).mean().reindex(MODELS) for k in SETS})
    lo = 0 if M.values.min() < 1 else float(np.nanpercentile(M.values, 2))  # lengths: scale from the smallest value
    heat(ax, M.values, MODELS, SETS, SEQ, vmin=lo, vmax=float(np.nanpercentile(M.values, 98)) or 1,
         fmt="{:.2f}" if M.values.max() < 10 else "{:.1f}", cbar=False)
    ax.set_title(f"{title}  ({col})", loc="left", fontsize=10)
for ax in axes.ravel()[len(HABITS):]: ax.axis("off")
plt.tight_layout(); plt.show()
print("habit columns used:", HABITS)
''')
md("""
**What this shows:**
- **Family rules hold everywhere:** Gemini never uses em dashes; Claude, GLM and Gemini never use curly quotes; Claude and GLM use the most em dashes. The rows are evenly coloured for these.
- **Curly quotes in GPT:** 2 to 4 times more common outside (for example gpt-5.6-terra 0.01 in Kaggle, 0.04 outside). A habit the external data exaggerates.
- **Semicolons:** kimi-k3 uses about twice as many in Kaggle (0.13) as outside (0.04 to 0.08); gpt-5.5 too (0.12 vs 0.07 to 0.09).
- **Sentence length:** every GPT model writes **longer sentences in Kaggle**, by 3 to 5 words (Kaggle 21 to 26, outside 18 to 22), and so does kimi-k3 (26 vs 21 to 23). That is a consistent change in setting, not noise.
- **Word length:** hostlike uses shorter words for every model except gpt-6-astra (about 5.0 vs 5.3 to 5.9), consistent with simpler topics.
""")

# ---------------------------------------------------------------- 4. drift heatmap
md("""
## 4. Which features drift, and for which models?

For every model and feature: **(external mean minus Kaggle mean) divided by the spread** (the standardised difference). 0 (grey) means the same, plus or minus 0.5 is a clear shift, plus or minus 1 is large. The 30 features shown are the ones that drift most on average across the three external sets.

For scale: with 150 texts on each side, pure chance alone gives differences of about plus or minus 0.1, so only cells with clear colour matter.
""")
code(r'''
def smd(a, b):
    return (b.mean(0) - a.mean(0)) / (np.sqrt((a.var(0) + b.var(0)) / 2) + 1e-9)
DR = {}
for k in EXT:
    DR[k] = pd.DataFrame({m: smd(F["Kaggle"].loc[S["Kaggle"].model == m, VARY].to_numpy(),
                                 F[k].loc[S[k].model == m, VARY].to_numpy()) for m in MODELS}, index=VARY)
top = pd.concat([DR[k].abs().mean(1) for k in EXT], axis=1).mean(1).sort_values(ascending=False).head(30).index
fig, axes = plt.subplots(1, 3, figsize=(22, 11), sharey=True)
for ax, k in zip(axes, EXT):
    heat(ax, DR[k].loc[top].values, list(top), MODELS, DIV, norm=TwoSlopeNorm(0, -1.2, 1.2), annot=False,
         cbar=(k == EXT[-1]), label="standardised difference (external - Kaggle)")
    ax.set_title(f"{k} vs Kaggle", loc="left")
plt.tight_layout(); plt.show()
summ = pd.DataFrame({k: {"features x models with |diff| > 0.5 (%)": 100 * (DR[k].abs() > 0.5).values.mean(),
                         "mean |diff|": DR[k].abs().values.mean()} for k in EXT}).T
print(summ.round(2).to_string())
''')
md("""
**What this shows:**
- **The two pools are close to Kaggle:** only about 1.5% of the (feature, model) cells shift by more than 0.5. **Hostlike drifts three times as much** (4.6%).
- **Horizontal bands** (a feature that shifts the same way for every model) point to a dataset-wide change, most likely the topics: outside texts have more capital letters, digits and title-case runs (names and numbers), fewer "we/us/our", and fewer "not X but Y".
- **Blocks under one family** are model-specific drift: GPT outside has more periods and shorter sentences (the blue `sent_len` rows under the GPT columns), and Gemini uses more hyphens outside.
- **Hostlike** has strong bands in word length and readability (`flesch`, `mean_syllables`, `long_word_share`): its prompts ask for simpler, everyday writing.
""")

# ---------------------------------------------------------------- 5. map
md("""
## 5. Where do the texts sit in "style space"?

All 290 style features squeezed into 2 dimensions (PCA, fitted on Kaggle only). Each panel is a density map: **darker = more texts there**. Same axes in every panel, so you can compare the shapes directly. The red line outlines where Kaggle is densest (half its peak density), drawn on every panel for reference. Caution: 2 axes keep only a small part of the 290 features, so this is a rough picture; sections 4 and 7 are the precise ones.
""")
code(r'''
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
sc = StandardScaler().fit(TX(F["Kaggle"][VARY]))
pca = PCA(2, random_state=0).fit(sc.transform(TX(F["Kaggle"][VARY])))
P = {k: pca.transform(sc.transform(TX(F[k][VARY]))) for k in SETS}
lim = np.percentile(np.vstack(list(P.values())), [1, 99], axis=0)
from scipy.ndimage import gaussian_filter
def dens(p):  # smoothed 2-D histogram, so the shapes are readable instead of grainy
    h, xe, ye = np.histogram2d(*p.T, bins=50, range=[lim[:, 0], lim[:, 1]], density=True)
    return gaussian_filter(h, 2.0), xe, ye
H0, xe, ye = dens(P["Kaggle"])
xc, yc = (xe[:-1] + xe[1:]) / 2, (ye[:-1] + ye[1:]) / 2
vmax = max(dens(P[k])[0].max() for k in SETS)
fig, axes = plt.subplots(1, 4, figsize=(20, 5.2), sharex=True, sharey=True)
for ax, k in zip(axes, SETS):
    H = dens(P[k])[0]
    ax.contourf(xc, yc, H.T, levels=np.linspace(vmax * 0.03, vmax, 12), cmap=SEQ, extend="neither")
    ax.contour(xc, yc, H0.T, levels=[H0.max() * 0.5], colors="#e34948", linewidths=1.5)
    ax.set_title(k, loc="left"); ax.set_xlabel("style axis 1")
axes[0].set_ylabel("style axis 2")
plt.tight_layout(); plt.show()
print("share of style variation the 2 axes keep:", pca.explained_variance_ratio_.round(3))
''')
md("""
**What this shows:**
- **Both pools cover the same area as Kaggle**: their dense region sits inside the red Kaggle outline.
- **Hostlike is shifted down and to the right**, and its peak sits near the lower edge of Kaggle's dense region. That is the shorter-words, easier-reading shift from section 4.
- Remember that this map keeps only about 7% of the style variation, so it shows the big shifts only.
""")

# ---------------------------------------------------------------- 6. who resembles whom
md("""
## 6. Who resembles whom?

For each dataset: how similar every model's average style is to every other model's (cosine similarity of the model averages, each measured from its own dataset's average, so a shift of a whole dataset does not count). Dark squares along the diagonal = families. The last row shows **external minus Kaggle**: orange means two models look more alike outside than in Kaggle, blue means less alike.
""")
code(r'''
z = StandardScaler().fit(F["Kaggle"][VARY])
def sim(k):
    c = pd.DataFrame(z.transform(F[k][VARY]), index=S[k].model.values).groupby(level=0).mean().reindex(MODELS).to_numpy()
    c = c - c.mean(0)  # centre on this dataset's own average: an overall shift of a dataset is not "models alike"
    c = c / np.linalg.norm(c, axis=1, keepdims=True)
    return c @ c.T
SIM = {k: sim(k) for k in SETS}
fig, axes = plt.subplots(2, 4, figsize=(26, 13))
for ax, k in zip(axes[0], SETS):
    heat(ax, SIM[k], MODELS, MODELS, SEQ, vmin=-0.2, vmax=1, annot=False, cbar=(k == SETS[-1]), label="similarity")
    ax.set_title(k, loc="left")
axes[1, 0].axis("off")
iu = np.triu_indices(len(MODELS), 1)
for ax, k in zip(axes[1, 1:], EXT):
    heat(ax, SIM[k] - SIM["Kaggle"], MODELS, MODELS, DIV, norm=TwoSlopeNorm(0, -0.6, 0.6), annot=False,
         cbar=(k == EXT[-1]), label="external - Kaggle")
    ax.set_title(f"{k} minus Kaggle  (match {np.corrcoef(SIM[k][iu], SIM['Kaggle'][iu])[0, 1]:.2f})", loc="left")
plt.tight_layout(); plt.show()
''')
md("""
**What this shows:**
- **The family map carries over:** the dark family squares (Claude, Gemini, GPT) are in every dataset, and the overall match is 0.88 (Pool 92k), 0.87 (Pool 60k) and 0.78 (hostlike).
- **Two Kaggle look-alikes weaken outside**, and both match our known mistakes:
  - In Kaggle, **GLM looks like Claude** (dark cells where the GLM rows meet the Claude columns, bottom-left, mirrored top-right). In **Pool 92k** that link is much weaker (blue in its "minus" panel); in Pool 60k and hostlike it stays.
  - In Kaggle, **Kimi looks like GPT**. In all three external sets much less (blue Kimi-GPT cells), while Kimi looks more like Gemini outside (orange).
- So the Kimi/GPT resemblance is missing from all external data, and the Claude/GLM one from Pool 92k, the first-stage data of our main model. Only Kaggle's own texts show them fully.
""")

# ---------------------------------------------------------------- 7. sibling clues
md("""
## 7. Do the clues that separate sibling models carry over?

This is the question that matters most for our score. Our worst mistakes are between **siblings** (models of the same family). For every sibling pair:
- **Kaggle separation**: how different the two models are in Kaggle (average absolute standardised difference over all style features). The **noise floor** column is the same number after shuffling the two models' labels: what pure chance gives with 150 texts each. If separation is close to the noise floor, our style features cannot tell the pair apart even in Kaggle.
- **Clue match** for each external set: the correlation between the pair's difference profile in Kaggle and outside. 1 = the same features separate them in the same direction; 0 = unrelated.
- **Kaggle vs Kaggle** is the fair ceiling: the same correlation between two halves of Kaggle (split by prompt, 75 texts per model in each half, averaged over 10 splits). Even Kaggle cannot match itself better than this, so an external set close to it carries the clues as well as can be measured. With half the texts, this ceiling is a little lower than a full-size comparison would be.
""")
code(r'''
rng = np.random.default_rng(0)
K = F["Kaggle"][VARY].to_numpy(); KM = S["Kaggle"].model.to_numpy()
KG = S["Kaggle"].grp.to_numpy(); UG = np.unique(KG)
HALVES = [np.isin(KG, rng.permutation(UG)[:len(UG) // 2]) for _ in range(10)]  # prompt halves of Kaggle
rows = []
for i, a in enumerate(MODELS):
    for b in MODELS[i + 1:]:
        if fam(a) != fam(b): continue
        ka, kb = K[KM == a], K[KM == b]
        t = smd(ka, kb)
        both = np.vstack([ka, kb]); noise = []
        for _ in range(20):
            p = rng.permutation(len(both)); noise.append(np.abs(smd(both[p[:len(ka)]], both[p[len(ka):]])).mean())
        r = {"pair": f"{a} | {b}", "Kaggle separation": np.abs(t).mean(), "noise floor": np.mean(noise)}
        r["clue match Kaggle vs Kaggle"] = np.mean([np.corrcoef(smd(K[(KM == a) & h], K[(KM == b) & h]),
                                                                smd(K[(KM == a) & ~h], K[(KM == b) & ~h]))[0, 1]
                                                    for h in HALVES])
        for k in EXT:
            E = F[k][VARY].to_numpy(); EM = S[k].model.to_numpy()
            r[f"clue match {k}"] = np.corrcoef(t, smd(E[EM == a], E[EM == b]))[0, 1]
        rows.append(r)
SIB = pd.DataFrame(rows).set_index("pair")
SIB = SIB.sort_values("Kaggle separation")
fig, axes = plt.subplots(1, 2, figsize=(15, 11), gridspec_kw={"width_ratios": [2, 3]})
heat(axes[0], SIB[["Kaggle separation", "noise floor"]].values, list(SIB.index), ["Kaggle separation", "noise floor"],
     SEQ, vmin=0.05, vmax=0.2, label="mean |standardised difference|")
axes[0].set_title("How different are the two siblings in Kaggle?", loc="left")
cm = [c for c in SIB if c.startswith("clue")]
heat(axes[1], SIB[cm].values, list(SIB.index), [c.replace("clue match ", "") for c in cm], SEQ, vmin=0, vmax=0.85,
     label="clue match (correlation)")
axes[1].set_yticklabels([])
axes[1].set_title("Do the same clues separate them outside?", loc="left")
plt.tight_layout(); plt.show()
print(SIB.round(3).to_string())
''')
md("""
**What this shows:**
- **Left:** for the top 7 pairs (glm-5.3 vs glm-5.3-flash, gemini-3.7 vs 3.8, gemini-3.6 vs 3.7 and 3.8, the gpt-5.6 luna / sol / terra trio), the Kaggle separation **is at the noise floor**: averaged over all 290 features, these siblings differ no more than chance. An average can hide a few real clues, and the Kaggle vs Kaggle column shows some exist: 0.22 to 0.32 for most of these pairs, but only 0.10 for glm-5.3 vs glm-5.3-flash. So their style clues are **weak**, and for the GLM pair almost absent, even inside Kaggle.
- **Right, hard pairs:** the external sets match about as well as Kaggle matches itself (for example gemini-3.7 vs 3.8: Kaggle vs Kaggle 0.22, external 0.10 to 0.26; gpt-5.6 luna vs terra: 0.30, external 0.38 to 0.43). The few weak clues these pairs have do appear outside. The values just above the ceiling come from it using half the texts.
- **Right, where clues are lost:** gemini-3.1-pro vs the flashes (Kaggle vs Kaggle 0.61 to 0.73; Pool 92k 0.42 to 0.57, hostlike 0.21 to 0.44) and gpt-6-astra vs the gpt-5.6 models (0.65 to 0.70; external 0.32 to 0.55). These two models are also the biggest losers in section 8.
- **Claude pairs** carry over at or near the ceiling in all three sets.
- **Practical meaning:** the hardest confusions (GLM pair, gemini 3.7/3.8, the gpt-5.6 trio) have weak style clues even in Kaggle, and the external data already carries what little there is. More data of the same kind cannot add a clue that is missing. The clues external data does lose belong to gemini-3.1-pro and gpt-6-astra.
""")

# ---------------------------------------------------------------- 8. teach test
md("""
## 8. Does the external data teach what Kaggle's test needs?

A direct test, using only our style features and a simple classifier (logistic regression):
- **Taught by Kaggle**: trained on Kaggle, scored on Kaggle prompts it never saw (5 folds, split by prompt, so it trains on about 120 texts per model each time).
- **Taught by X**: trained only on 120 texts per model from external set X, scored on all of Kaggle. Repeated with 5 different random samples of 120; the charts show the average and the title gives the spread.

Same classifier, same features, same number of training texts per model, so the only difference is where the training texts came from. Each heatmap cell is the share of Kaggle texts of that model (row) that were labelled as each model (column); a perfect classifier is a dark diagonal. The table after it shows the recall per model (diagonal) side by side.
""")
code(r'''
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
clf = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=3000))
Xk = TX(F["Kaggle"][VARY]).to_numpy(); yk = S["Kaggle"].model.to_numpy()
pk = np.empty(len(yk), dtype=object)
for tr_i, te_i in GroupKFold(5).split(Xk, yk, S["Kaggle"].grp):
    pk[te_i] = clf().fit(Xk[tr_i], yk[tr_i]).predict(Xk[te_i])
PRED = {"Kaggle": [pk]}
for k in EXT:
    PRED[k] = []
    for seed in range(5):  # 120 texts per model, 5 different samples
        idx = S[k].groupby("model", group_keys=False).apply(lambda g: g.sample(min(120, len(g)), random_state=seed)).index
        PRED[k].append(clf().fit(TX(F[k].loc[idx, VARY]).to_numpy(), S[k].loc[idx, "model"].to_numpy()).predict(Xk))
fig, axes = plt.subplots(1, 4, figsize=(28, 8), sharey=True)
REC, BACC = {}, {}
for ax, k in zip(axes, SETS):
    cmx = np.mean([confusion_matrix(yk, p, labels=MODELS, normalize="true") for p in PRED[k]], axis=0)
    REC[k] = np.diag(cmx)
    b = [100 * balanced_accuracy_score(yk, p) for p in PRED[k]]
    BACC[k] = (np.mean(b), np.std(b))
    heat(ax, cmx, MODELS, MODELS, SEQ, vmin=0, vmax=0.6, annot=False, cbar=(k == SETS[-1]), label="share of the row")
    ax.set_title(f"Taught by {k}: balanced accuracy {np.mean(b):.1f}%" + (f" (± {np.std(b):.1f})" if len(b) > 1 else ""),
                 loc="left")
plt.tight_layout(); plt.show()
fig, ax = plt.subplots(figsize=(7, 8))
R = pd.DataFrame(REC, index=MODELS)
heat(ax, R.values * 100, MODELS, [f"taught by {k}" for k in SETS], SEQ, vmin=0, vmax=70, fmt="{:.0f}",
     label="recall on Kaggle (%)")
ax.set_title("Recall per model on Kaggle texts (%)", loc="left"); plt.tight_layout(); plt.show()
print({k: f"{m:.1f} +- {s:.1f}" for k, (m, s) in BACC.items()})
print(R.mul(100).round(0).astype(int).to_string())
''')
md("""
**What this shows:**
- **Kaggle texts are worth more than external texts:** with 120 texts per model in every case, training on Kaggle gives **33.3%**, on Pool 92k **25.0%** (± 0.3 over 5 samples), Pool 60k **24.0%** (± 0.4), hostlike **23.4%** (± 0.6), always scored on Kaggle prompts the classifier never saw. About 8 to 10 points are lost purely from where the training texts came from.
- **Biggest losses** when taught by external data: gemini-3.1-pro (61% down to 37% to 46%), gpt-6-astra (51% down to 25% to 36%), kimi-k3 (27% down to 7% to 16%), glm-5.3 (26% down to 8% to 19%), gpt-5.6-luna (30% down to 17% to 18%).
- **A new confusion appears:** taught by external data, many Kaggle GPT texts are labelled **qwen3.8-max** (the lit qwen column in the GPT rows). Outside, Qwen writes more like Kaggle's GPT.
- Pool 92k is slightly ahead of the other two (about 1 point, more than its spread); Pool 60k and hostlike are within noise of each other.
- This is why stage 2 on Kaggle's own texts matters so much, and why the 3 to 4 points we lose on the hidden test are hard to win back with more outside data.
""")

# ---------------------------------------------------------------- 9. adversarial
md("""
## 9. Can a classifier tell the datasets apart?

Here a classifier gets mixed Kaggle and external texts (class-matched) and must guess **which dataset** each text came from. The score is AUC: **0.5 = it cannot tell (the data look the same)**, 1.0 = it is never wrong. The folds are split by prompt, so the classifier is never tested on a prompt it saw in training (otherwise it can recognise the prompt instead of the dataset). Pool 92k has no prompt ids, so its folds cannot be split this way and its numbers are an upper bound.
- **Style features**: our 290 hand-made style numbers only.
- **Character patterns**: 2 to 4 character snippets (TF-IDF), which also see the topic and vocabulary.
""")
code(r'''
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
cv = StratifiedGroupKFold(5, shuffle=True, random_state=0)
A = {}
for k in EXT:
    y = np.r_[np.zeros(len(S["Kaggle"])), np.ones(len(S[k]))]
    g = np.r_[S["Kaggle"].grp.to_numpy(), S[k].grp.to_numpy()]
    Xs = TX(pd.concat([F["Kaggle"][VARY], F[k][VARY]])).to_numpy()
    p1 = cross_val_predict(clf(), Xs, y, cv=cv, groups=g, method="predict_proba")[:, 1]
    Xc = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=3, max_features=100_000,
                         sublinear_tf=True).fit_transform(pd.concat([S["Kaggle"].text, S[k].text]))
    p2 = cross_val_predict(LogisticRegression(C=1.0, max_iter=2000), Xc, y, cv=cv, groups=g, method="predict_proba")[:, 1]
    A[k + (" (upper bound)" if k == "Pool 92k" else "")] = {"style features": roc_auc_score(y, p1),
                                                            "character patterns": roc_auc_score(y, p2)}
A = pd.DataFrame(A).T
print(A.round(3).to_string())
fig, ax = plt.subplots(figsize=(6, 3.6))
heat(ax, A.values, list(A.index), list(A.columns), SEQ, vmin=0.5, vmax=1.0, fmt="{:.3f}", label="AUC (0.5 = alike)")
ax.set_title("Kaggle vs external: how easy to tell apart", loc="left"); plt.tight_layout(); plt.show()
''')
md("""
**What this shows:**
- **Clearly, but not perfectly, distinguishable.** With style numbers only, AUC 0.76 to 0.77 for the pools and 0.83 for hostlike; with character patterns, 0.81 for the pools and 0.86 for hostlike.
- Splitting by prompt matters: with random folds the same test gave 0.82 to 0.88 and 0.94 to 0.97, because the classifier partly learned to recognise Kaggle's 150 prompts.
- That fits sections 4 and 5: no single feature differs much, but many small shifts add up. **Hostlike is the most different** of the three; the two pools are equally close.
""")

md("""
## Summary

| Question | Pool 92k | Pool 60k | Hostlike |
|---|---|---|---|
| Model mix like Kaggle? | uneven (opus-5, gemini-3.7, qwen short) | yes, balanced | uneven (GPT heavy, Gemini short) |
| Length like Kaggle? | yes (101 vs 101 words) | yes (100) | shorter (86) |
| Features drifting > 0.5 sd | 1.5% | 1.4% | 4.6% |
| Family map match | 0.88 | 0.87 | 0.78 |
| Tell-apart AUC, split by prompt (style / characters) | 0.76 / 0.81 (upper bound) | 0.77 / 0.81 | 0.83 / 0.86 |
| Classifier taught on it, scored on Kaggle | 25.0% | 24.0% | 23.4% (Kaggle-taught 33.3%) |

**In one line:** the external data looks like Kaggle on average and teaches the families correctly, but the details differ:
- GPT sentence length and Kimi semicolons;
- the 3.7 vs 3.8 length clue;
- the Kimi-like-GPT resemblance (all sets) and GLM-like-Claude (Pool 92k);
- the clues for gemini-3.1-pro and gpt-6-astra.

Our hardest siblings have weak style clues even in Kaggle, and the external data carries those about as well as can be measured. Kaggle's own texts remain the best teacher by about 8 to 10 points; Pool 92k is slightly the best external one.
""")

nb["cells"] = C
nb.metadata["kernelspec"] = {"name": "ntl-venv", "display_name": "Python (name_thy_llm .venv)", "language": "python"}
out = Path(__file__).with_name("data_compare.ipynb")
nbf.write(nb, out)
print("wrote", out)
