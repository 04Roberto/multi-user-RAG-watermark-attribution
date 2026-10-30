"""
Simplified knowledge-base poisoning test for the multi-user watermark.

For each of the 9 injected watermark documents (3 users x 3 facts) we:
  1. re-write the recorded watermark text into the recorded doc id (baseline state)
  2. poison a fraction of those documents with one of three attacks
  3. query the vector DB with the same question used at injection time
  4. count a fact as VERIFIED if its document is still in the top-k results
     AND still contains >= 60% of the original watermark words
  5. restore all documents and repeat (random choice of which docs are hit)

Attacks:
  overwrite : doc text replaced by unrelated corpus text
  dilute    : 3 unrelated corpus passages appended to the doc
  wordDrop  : 50% of words randomly deleted (crude rewrite)

This is NOT the embedding-perturbation attack from the threat-model framework.
It only tests whether the watermark facts survive plain edits to the stored
documents. No LLM is used; only Contriever + the Chroma vector DB.
"""
import os, sys, glob, json, random, argparse, re, csv

def words(s):
    return set(re.findall(r"[a-z0-9]+", s.lower()))

def overlap(orig, cur):
    o = words(orig)
    return len(o & words(cur)) / max(1, len(o))

def question(wm):
    return f"What is the relationship between {wm[0]} and {wm[1]} ?"

def atk_overwrite(text, filler, rng):
    return rng.choice(filler)

def atk_dilute(text, filler, rng):
    return text + " " + " ".join(rng.sample(filler, 3))

def atk_worddrop(text, filler, rng):
    w = text.split()
    kept = [x for x in w if rng.random() > 0.5]
    return " ".join(kept) if kept else ""

ATTACKS = {"overwrite": atk_overwrite, "dilute": atk_dilute, "wordDrop": atk_worddrop}

def load_facts(inj_dir):
    facts = []
    for f in sorted(glob.glob(os.path.join(inj_dir, "user_*_injected.json"))):
        uid = int(re.search(r"user_(\d+)_injected", f).group(1))
        for wm, ids, texts in json.load(open(f)):
            facts.append({"user": uid, "wm": wm, "id": ids[0], "text": texts[0]})
    return facts

def run_test(store, facts, filler, top_k, trials, fracs, seed, log=print):
    """Direct replace via collection.update (NOT update_context, which appends)."""
    rng = random.Random(seed)
    cur, snap = {}, {}

    def put(fact, text):
        e = store.get_embedding(text)
        store.collection.update(ids=[fact["id"]], embeddings=[e], documents=[text])
        cur[fact["id"]] = text

    def restore():
        for f in facts:
            put(f, snap[f["id"]])

    def verified(f):
        res = store.search_context(question(f["wm"]), top_k)
        if f["id"] not in res["ids"][0]:
            return False
        return overlap(f["text"], cur[f["id"]]) >= 0.6

    for f in facts:
        snap[f["id"]] = store.collection.get(ids=[f["id"]])["documents"][0]
        cur[f["id"]] = snap[f["id"]]
    users = sorted(set(f["user"] for f in facts))
    rows = []
    base = {u: sum(verified(f) for f in facts if f["user"] == u) for u in users}
    log(f"BASELINE verified per user (of 3): {base}")
    for name, atk in ATTACKS.items():
        for frac in fracs:
            k = round(frac * len(facts))
            tot = {u: 0.0 for u in users}
            for _ in range(trials):
                hit = rng.sample(facts, k)
                for f in hit:
                    put(f, atk(snap[f["id"]], filler, rng))
                for u in users:
                    tot[u] += sum(verified(f) for f in facts if f["user"] == u)
                restore()
            avg = {u: tot[u] / trials for u in users}
            rows.append({"attack": name, "fraction": frac, "docs_poisoned": k,
                         **{f"user{u}": round(avg[u], 2) for u in users},
                         "total": round(sum(avg.values()), 2)})
            log(f"{name:9s} frac={frac:.2f} poisoned={k}  surviving per user={ {u: round(avg[u],2) for u in users} }  total={sum(avg.values()):.2f}/{len(facts)}")
    restore()
    after = sum(verified(f) for f in facts)
    log(f"After restore: {after} facts verify (baseline {sum(base.values())})")
    return base, rows

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--eval_model_code", default="contriever")
    p.add_argument("--eval_dataset", default="nfcorpus")
    p.add_argument("--split", default="test")
    p.add_argument("--score_function", default="cosine")
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--inj_dir", default="output/multi_user/nfcorpus/validated_user_assignments")
    p.add_argument("--top_k", type=int, default=5)
    p.add_argument("--trials", type=int, default=5)
    p.add_argument("--seed", type=int, default=633)
    p.add_argument("--out_dir", default="output/multi_user/nfcorpus/poison_test")
    return p.parse_args()

if __name__ == "__main__":
    import transformers.utils.import_utils as import_utils
    import_utils.check_torch_load_is_safe = lambda: None
    root = os.path.abspath(os.path.dirname(__file__))
    sys.path.append(root); sys.path.append(os.path.join(root, "src"))
    import torch
    from rag.vectorstore import VectorStore, check_collection
    from src.utils import load_beir_datasets, load_models

    a = parse_args()
    torch.cuda.set_device(a.gpu_id)
    coll = f"{a.eval_dataset}_{a.eval_model_code}_{a.score_function}"
    model, c_model, tokenizer, get_emb = load_models(a.eval_model_code)
    corpus, queries, qrels = load_beir_datasets(a.eval_dataset, a.split)
    ok, n = check_collection(coll)
    if not (ok and n > 0):
        sys.exit("vector store missing")
    store = VectorStore(model, tokenizer, get_emb, corpus, "cuda", coll, use_local=True)

    facts = load_facts(a.inj_dir)
    print(f"Loaded {len(facts)} watermark facts from {a.inj_dir}")
    used = {f["id"] for f in facts}
    pool = [v["text"] for v in corpus.values() if v.get("text")]
    filler = [t[:600] for t in random.Random(a.seed).sample(pool, 200)]

    os.makedirs(a.out_dir, exist_ok=True)
    logf = open(os.path.join(a.out_dir, "poison_log.txt"), "w")
    def log(s):
        print(s); logf.write(s + "\n"); logf.flush()

    log(f"top_k={a.top_k} trials={a.trials} seed={a.seed}")
    base, rows = run_test(store, facts, filler, a.top_k, a.trials,
                          [0.0, 0.25, 0.5, 0.75, 1.0], a.seed, log)
    json.dump({"baseline": base, "rows": rows, "top_k": a.top_k, "trials": a.trials},
              open(os.path.join(a.out_dir, "poison_results.json"), "w"), indent=2)
    with open(os.path.join(a.out_dir, "poison_results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    log(f"Saved to {a.out_dir}. Watermark documents restored to injected state.")
