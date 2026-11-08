"""Hand-made stylometric features for LLM authorship (the "tabular" view of a passage).

Pure Python (re + math only) so the exact same code can be pasted into the Kaggle notebook.
extract(text) -> list of floats, aligned with NAMES. Rates are per 100 characters or per 100 words,
so passages of different length stay comparable.

Why these: LLMs differ in punctuation habits (em dashes, curly quotes, semicolons), sentence rhythm,
function-word and connective use ("however", "moreover"), list and markdown habits, and pet words.
"""
import math
import re

PUNCT = [",", ".", ";", ":", "!", "?", "—", "–", "-", "(", ")", '"', "'", "“", "”", "‘",
         "’", "…", "*", "#", "/", "%", "$", "&", "@", "[", "]", "{", "}", "_", "~", "`", "|", "=", "+",
         "<", ">", "\\", "→", "•", "·", "\n"]
PUNCT_NAME = {",": "comma", ".": "period", ";": "semicolon", ":": "colon", "!": "exclaim", "?": "question",
              "—": "emdash", "–": "endash", "-": "hyphen", "(": "lparen", ")": "rparen", '"': "dquote",
              "'": "squote", "“": "ldquo", "”": "rdquo", "‘": "lsquo", "’": "rsquo",
              "…": "ellipsis", "*": "star", "#": "hash", "/": "slash", "%": "percent", "$": "dollar",
              "&": "amp", "@": "at", "[": "lbracket", "]": "rbracket", "{": "lbrace", "}": "rbrace",
              "_": "underscore", "~": "tilde", "`": "backtick", "|": "pipe", "=": "equals", "+": "plus",
              "<": "lt", ">": "gt", "\\": "backslash", "→": "arrow", "•": "bullet", "·": "middot",
              "\n": "newline"}

# function words, connectives and hedges (rate per 100 words)
WORDS = sorted(set("""the of and a to in is that it as with for was on are by this but not or from at an be have which
you we i they their its can more than also while however moreover furthermore thus therefore yet even just still
only every each both between through across within without into over under about these those there when where who
whose what how why if so such like often rather perhaps ultimately indeed whether might may could would should must
will does do has had been being one all some many most other any no never always quietly simply truly really
essentially fundamentally arguably remarkably notably crucially additionally ultimately tapestry delve testament
landscape realm nuanced intricate crucial pivotal foster navigate underscore embrace vibrant journey beacon
symphony whisper echo fascinating profoundly seamless robust leverage human world time life people way
something nothing everything here now then first second third finally instead whereas although though because
since unless until once again already often sometimes usually less little much very quite almost nearly""".split()))

STARTERS = ["the", "it", "in", "this", "but", "and", "a", "we", "i", "they", "when", "if", "as", "for", "one", "what",
            "there", "yet", "so", "still", "perhaps", "not", "no", "that", "these", "to", "at", "on", "by", "with"]

RE_WORD = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)?")  # unicode letters, so "cafe" with an accent stays one word
RE_SENT = re.compile(r"(?<=[.!?])[\"'”’)]*\s+(?=[\"'“‘(\[]*[A-Z0-9])")
RE_VOWELS = re.compile(r"[aeiouy]+")
PATTERNS = {
    "not_just": re.compile(r"\bnot (?:just|only|merely|simply)\b", re.I),
    "not_but": re.compile(r"\bnot\b[^.!?]{0,50}\bbut\b", re.I),
    "its_not": re.compile(r"\b(?:it|that|this)(?:['’]s| is) not\b", re.I),
    "contraction": re.compile(r"\w['’](?:t|s|re|ve|ll|d|m)\b", re.I),
    "oxford": re.compile(r",\s+(?:and|or)\s", re.I),
    "ly_adv": re.compile(r"\b\w{3,}ly\b", re.I),
    "ing": re.compile(r"\b\w{3,}ing\b", re.I),
    "of_the": re.compile(r"\bof the\b", re.I),
    "spaced_emdash": re.compile(r" — "),
    "tight_emdash": re.compile(r"\w—\w"),
    "spaced_hyphen": re.compile(r" - "),
    "double_hyphen": re.compile(r"--"),
    "three_dots": re.compile(r"\.\.\."),
    "double_space": re.compile(r"  "),
    "space_before_punct": re.compile(r" [,.;:!?]"),
    "no_space_after_period": re.compile(r"\.[A-Z]"),
    "bold": re.compile(r"\*\*[^*]+\*\*"),
    "bullet_line": re.compile(r"(?m)^\s*(?:[-*•]|\d+[.)])\s"),
    "header_line": re.compile(r"(?m)^#{1,6}\s"),
    "digits": re.compile(r"\d"),
    "number": re.compile(r"\b\d[\d,.]*\b"),
    "year": re.compile(r"\b(?:1[5-9]|20)\d\d\b"),
    "quote_pair": re.compile(r"[\"“][^\"”]{1,200}[\"”]"),
    "paren_group": re.compile(r"\([^)]{1,200}\)"),
    "colon_list": re.compile(r":\s"),
    "semi_clause": re.compile(r";\s"),
    "rhetorical_q": re.compile(r"\?"),
    "we_us_our": re.compile(r"\b(?:we|us|our|ours)\b", re.I),
    "i_me_my": re.compile(r"\b(?:i|me|my|mine)\b", re.I),
    "you_your": re.compile(r"\b(?:you|your|yours)\b", re.I),
    "all_caps": re.compile(r"\b[A-Z]{2,}\b"),
    "title_case_run": re.compile(r"\b[A-Z][a-z]+(?: [A-Z][a-z]+){1,}\b"),
}


def _stats(vals):
    n = len(vals)
    if not n:
        return 0.0, 0.0, 0.0, 0.0
    m = sum(vals) / n
    sd = math.sqrt(sum((v - m) ** 2 for v in vals) / n)
    return m, sd, float(max(vals)), float(min(vals))


def _build_names():
    names = ["chars", "words", "sentences", "log_chars", "mean_word_len", "std_word_len", "long_word_share",
             "very_long_word_share", "sent_len_mean", "sent_len_std", "sent_len_max", "sent_len_min", "sent_len_cv",
             "short_sent_share", "long_sent_share", "last_sent_len", "first_sent_len", "ttr", "hapax_share",
             "mean_syllables", "flesch", "upper_char_share", "cap_word_share", "nonascii_share", "alpha_share",
             "space_share", "starts_quote", "ends_period", "ends_question", "ends_quote", "ends_paren",
             "first_word_len", "comma_per_sent", "comma_per_word", "words_per_comma_clause"]
    names += [f"p_{PUNCT_NAME[c]}" for c in PUNCT]
    names += [f"pat_{k}" for k in PATTERNS]
    names += [f"w_{w}" for w in WORDS]
    names += [f"start_{w}" for w in STARTERS]
    names += ["curly_quote_share", "emdash_per_sent", "sent_starts_conj_share", "sent_starts_the_share"]
    return names


NAMES = _build_names()


def extract(text):
    n_chars = max(len(text), 1)
    words = RE_WORD.findall(text)
    lw = [w.lower() for w in words]
    nw = max(len(words), 1)
    sents = [s for s in RE_SENT.split(text.strip()) if s.strip()] or [text]
    ns = len(sents)
    sent_lens = [len(RE_WORD.findall(s)) for s in sents]
    wl = [len(w) for w in words]
    mwl, swl, _, _ = _stats(wl)
    sm, ssd, smax, smin = _stats(sent_lens)
    counts = {}
    for w in lw:
        counts[w] = counts.get(w, 0) + 1
    syll = [max(1, len(RE_VOWELS.findall(w))) for w in lw]
    msyl = sum(syll) / nw
    flesch = 206.835 - 1.015 * (nw / ns) - 84.6 * msyl
    upper = sum(1 for c in text if c.isupper())
    alpha = sum(1 for c in text if c.isalpha())
    nonascii = sum(1 for c in text if ord(c) > 127)
    spaces = text.count(" ")
    stripped = text.rstrip()
    last = stripped[-1:] if stripped else ""
    caps_mid = sum(1 for w in words[1:] if w[0].isupper())
    commas = text.count(",")

    f = [float(n_chars), float(len(words)), float(ns), math.log(n_chars), mwl, swl,
         sum(1 for x in wl if x >= 7) / nw, sum(1 for x in wl if x >= 10) / nw,
         sm, ssd, smax, smin, (ssd / sm if sm else 0.0),
         sum(1 for x in sent_lens if x < 8) / ns, sum(1 for x in sent_lens if x > 30) / ns,
         float(sent_lens[-1]), float(sent_lens[0]),
         len(counts) / nw, sum(1 for v in counts.values() if v == 1) / nw,
         msyl, flesch, upper / n_chars, caps_mid / nw, nonascii / n_chars, alpha / n_chars, spaces / n_chars,
         1.0 if text[:1] and text[:1] in "\"'“‘" else 0.0, 1.0 if last == "." else 0.0, 1.0 if last == "?" else 0.0,
         1.0 if last and last in "\"”'’" else 0.0, 1.0 if last == ")" else 0.0,
         float(len(words[0])) if words else 0.0, commas / ns, commas / nw, nw / (commas + ns)]
    f += [text.count(c) * 100.0 / n_chars for c in PUNCT]
    f += [len(p.findall(text)) * 100.0 / (n_chars if k in ("digits", "double_space", "space_before_punct",
                                                              "no_space_after_period", "three_dots") else nw)
          for k, p in PATTERNS.items()]
    f += [counts.get(w, 0) * 100.0 / nw for w in WORDS]
    starts = [(RE_WORD.findall(s) or [""])[0].lower() for s in sents]
    f += [starts.count(w) / ns for w in STARTERS]
    curly = sum(text.count(c) for c in "“”‘’")
    straight = text.count('"') + text.count("'")
    f += [curly / (curly + straight) if curly + straight else -1.0, text.count("—") / ns,
          sum(1 for w in starts if w in ("but", "and", "yet", "so", "or")) / ns, starts.count("the") / ns]
    return f


if __name__ == "__main__":
    demo = "Language shapes thought—or so the story goes. But does it? “Not always,” she said; however, it often does."
    v = extract(demo)
    assert len(v) == len(NAMES), (len(v), len(NAMES))
    print(len(NAMES), "features; sample:", {k: round(x, 3) for k, x in zip(NAMES, v) if x and k.startswith(("p_em", "p_semi", "words", "sent"))})
