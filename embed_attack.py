"""
Simplified embedding-perturbation attack on the stored watermark documents.

Gray-box attacker: edits the STORED VECTOR of a watermark document, never its text.
  e' = e + eps * |e| * u      (u = unit direction, |e| = norm of the stored vector)
  directed : u points away from the verification query (orthogonal part of -q)
  random   : u is a random direction (control)
eps is raised step by step (the retry loop, 8 rounds, framework allows up to 10).

Measured per eps:
  - how many of the 9 watermark facts still verify (doc still in top-k for its query)
  - per user counts
  - cosine(e, e')            : how far the vector moved (stealth)
  - overlap of top-k for 20 ordinary nfcorpus queries (collateral damage)
Original vectors are restored at the end.

Not the gradient-optimised attack through Contriever. No LLM is used.
"""
import os, sys, json, csv, random, argparse
import numpy as np
from poison_test import load_facts, question

def emb(store, t):
    return np.asarray(store.get_embedding(t), dtype=float).reshape(-1)

def set_vec(store, i, v):
    if v.ndim != 1 or v.size < 16 or not np.all(np.isfinite(v)):
        raise RuntimeError(f"refusing to write bad vector for {i}: shape={v.shape}")
    store.collection.update(ids=[i], embeddings=[v.tolist()])

def reset_docs(store, facts):
    """Read the stored vectors as they are (original passage + watermark). No text change."""
    orig = {}
    ids = [f["id"] for f in facts]
    res = store.collection.get(ids=ids, include=["embeddings"])
    embs = res.get("embeddings")
    print(f"stored-embedding check: type={type(embs).__name__} ids={res['ids']}", flush=True)
    if embs is None or len(embs) != len(res["ids"]):
        raise SystemExit("could not read stored embeddings from Chroma (got None / wrong count). Nothing was changed.")
    for i, v in zip(res["ids"], embs):
        v = np.asarray(v, dtype=float).reshape(-1)
        if v.size < 16 or not np.all(np.isfinite(v)):
            raise SystemExit(f"stored embedding for {i} is bad: size={v.size}. Nothing was changed.")
        orig[i] = v
    return orig

def direction(mode, e, qv, rng):
    eh = e / np.linalg.norm(e)
    d = -(qv / np.linalg.norm(qv)) if mode == "directed" else rng.normal(size=e.shape)
    d = d - (d @ eh) * eh
    return d / np.linalg.norm(d)

def retrieved(store, f, top_k):
    return f["id"] in store.search_context(question(f["wm"]), top_k)["ids"][0]

def run_attack(store, facts, normal_qs, eps_grid, trials, top_k, seed, log=print):
    rng = np.random.default_rng(seed)
    orig = reset_docs(store, facts)
    users = sorted({f["user"] for f in facts})
    qvec = {f["id"]: emb(store, question(f["wm"])) for f in facts}
    for f in facts[:2]:
        a, b = orig[f["id"]], qvec[f["id"]]
        print(f"DEBUG {f['id']}: doc vec shape={a.shape} norm={np.linalg.norm(a):.4f} first3={a[:3]} | query vec shape={b.shape} norm={np.linalg.norm(b):.4f} first3={b[:3]}", flush=True)
    bad = [f["id"] for f in facts if np.linalg.norm(orig[f["id"]]) < 1e-6 or np.linalg.norm(qvec[f["id"]]) < 1e-6]
    if bad:
        raise SystemExit(f"zero-length vectors for {bad}. Nothing was changed.")
    base_n = [set(store.search_context(q, top_k)["ids"][0]) for q in normal_qs]
    base = {u: sum(retrieved(store, f, top_k) for f in facts if f["user"] == u) for u in users}
    log(f"BASELINE verified per user (of 3): {base}  total={sum(base.values())}/9")
    rows, first_drop = [], {}
    for mode in ["directed", "random"]:
        for eps in eps_grid:
            tot = {u: 0.0 for u in users}; ov = 0.0; cos = 0.0
            for t in range(trials):
                for f in facts:
                    e = orig[f["id"]]
                    u_ = direction(mode, e, qvec[f["id"]], rng)
                    e2 = e + eps * np.linalg.norm(e) * u_
                    cos += (e @ e2) / (np.linalg.norm(e) * np.linalg.norm(e2)) / len(facts) / trials
                    set_vec(store, f["id"], e2)
                ok = {f["id"]: retrieved(store, f, top_k) for f in facts}
                for f in facts:
                    tot[f["user"]] += ok[f["id"]]
                    if mode == "directed" and not ok[f["id"]]:
                        first_drop.setdefault(f["id"], eps)
                ov += np.mean([len(base_n[i] & set(store.search_context(q, top_k)["ids"][0])) / top_k
                               for i, q in enumerate(normal_qs)]) / trials
                for f in facts: set_vec(store, f["id"], orig[f["id"]])
            avg = {u: tot[u] / trials for u in users}
            row = {"mode": mode, "eps": eps, "cosine": round(cos, 3),
                   **{f"user{u}": round(avg[u], 2) for u in users},
                   "total": round(sum(avg.values()), 2), "normal_query_overlap": round(ov, 3)}
            rows.append(row)
            log(f"{mode:8s} eps={eps:<5} cos={cos:.3f} surviving={ {u: round(avg[u],2) for u in users} } total={sum(avg.values()):.2f}/9  normal-query overlap={ov:.2f}")
    after = sum(retrieved(store, f, top_k) for f in facts)
    log(f"After restore: {after}/9 verify (baseline was {sum(base.values())}/9)")
    log("First eps where each fact dropped (directed): " + json.dumps(first_drop))
    return base, rows, first_drop

if __name__ == "__main__":
    import transformers.utils.import_utils as iu
    iu.check_torch_load_is_safe = lambda: None
    root = os.path.abspath(os.path.dirname(__file__))
    sys.path.append(root); sys.path.append(os.path.join(root, "src"))
    import torch
    from rag.vectorstore import VectorStore, check_collection
    from src.utils import load_beir_datasets, load_models
    p = argparse.ArgumentParser()
    p.add_argument("--top_k", type=int, default=5)
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--seed", type=int, default=633)
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--inj_dir", default="output/multi_user/nfcorpus/validated_user_assignments")
    p.add_argument("--out_dir", default="output/multi_user/nfcorpus/embed_attack")
    a = p.parse_args()
    torch.cuda.set_device(a.gpu_id)
    coll = "nfcorpus_contriever_cosine"
    model, c_model, tok, get_emb = load_models("contriever")
    corpus, queries, qrels = load_beir_datasets("nfcorpus", "test")
    ok, n = check_collection(coll)
    if not (ok and n > 0): sys.exit("vector store missing")
    store = VectorStore(model, tok, get_emb, corpus, "cuda", coll, use_local=True)
    facts = load_facts(a.inj_dir)
    normal = random.Random(a.seed).sample(list(queries.values()), 20)
    os.makedirs(a.out_dir, exist_ok=True)
    lf = open(os.path.join(a.out_dir, "embed_log.txt"), "w")
    def log(s): print(s, flush=True); lf.write(s + "\n"); lf.flush()
    log(f"top_k={a.top_k} trials={a.trials} facts={len(facts)}")
    base, rows, fd = run_attack(store, facts, normal, [0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5],
                                a.trials, a.top_k, a.seed, log)
    json.dump({"baseline": base, "rows": rows, "first_drop": fd}, open(os.path.join(a.out_dir, "embed_results.json"), "w"), indent=2)
    with open(os.path.join(a.out_dir, "embed_results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    log("Saved. Original vectors restored.")
