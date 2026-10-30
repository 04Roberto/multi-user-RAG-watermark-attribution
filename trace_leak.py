"""
trace_leak.py - given a leaked text, score every user two ways and say who is accused.
  1. green-word score (z) using each user's green-list key            -> accused if z > --z_thr
  2. fact check: how many of the user's assigned watermark facts (both entities) appear in the text -> accused if >= --tau
Run from the repo root (so src/ and output/ are found):
  python trace_leak.py --text_file leak.txt --model_path <HF path> --users 1 2 3
"""
import argparse, glob, json, os, re, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from user_greenlist import make_user_key, green_key, detect


def find_triples(obj, out):
    """Find every [e1, e2, relation] (list of 3 strings) anywhere inside a JSON structure."""
    if isinstance(obj, list):
        if len(obj) == 3 and all(isinstance(x, str) for x in obj): out.append(obj)
        else:
            for x in obj: find_triples(x, out)
    elif isinstance(obj, dict):
        for x in obj.values(): find_triples(x, out)


def user_facts(uid, pattern):
    files = glob.glob(pattern.format(uid=uid), recursive=True)
    triples = []
    for f in files[:1]:
        find_triples(json.load(open(f)), triples)
    return triples, (files[0] if files else None)


def fact_hits(text, triples):
    t = text.lower()
    has = lambda e: re.search(r"(?<!\w)" + re.escape(e.lower()) + r"(?!\w)", t) is not None   # whole word/phrase only
    return sum(1 for e1, e2, _ in triples if has(e1) and has(e2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--text_file", required=True)
    ap.add_argument("--model_path", help="only the tokenizer is loaded (no GPU needed)")
    ap.add_argument("--users", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--vocab_size", type=int, default=32000)
    ap.add_argument("--z_thr", type=float, default=4.0)
    ap.add_argument("--tau", type=int, default=2)
    ap.add_argument("--assign_glob", default="output/multi_user/**/user_{uid}_validated_assignment.json")
    a = ap.parse_args()
    text = open(a.text_file).read()
    if not a.model_path: sys.exit("need --model_path (tokenizer) for the green-word score")
    from transformers import AutoTokenizer
    ids = AutoTokenizer.from_pretrained(a.model_path)(text, add_special_tokens=False).input_ids
    print(f"text: {len(ids)} tokens\n")
    print(f"{'user':<6}{'green z':>9}{'facts found':>13}   verdict")
    for u in a.users:
        z, g, n = detect(ids, green_key(make_user_key(u)), a.vocab_size)
        triples, path = user_facts(u, a.assign_glob.replace("{uid}", str(u)))
        hits = fact_hits(text, triples) if triples else None
        flags = []
        if z > a.z_thr: flags.append("green words")
        if hits is not None and hits >= a.tau: flags.append("facts")
        hs = "n/a" if hits is None else f"{hits}/{len(triples)}"
        print(f"user{u:<2}{z:9.2f}{hs:>13}   {'ACCUSED (' + ' + '.join(flags) + ')' if flags else '-'}")
    print("\nAccused if green z > %.1f or at least %d of the user's facts appear. Fix the thresholds BEFORE looking at results." % (a.z_thr, a.tau))
