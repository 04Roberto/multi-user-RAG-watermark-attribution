"""Recompute every stored vector from the stored document text (fixes all-zero DB)."""
import os, sys
import numpy as np

def reembed(store, bs=64, log=print):
    ids = store.collection.get(include=[])["ids"]
    log(f"re-embedding {len(ids)} documents")
    for i in range(0, len(ids), bs):
        b = ids[i:i + bs]
        res = store.collection.get(ids=b, include=["documents"])
        embs = [np.asarray(store.get_embedding(d), dtype=float).reshape(-1).tolist() for d in res["documents"]]
        if any(not np.all(np.isfinite(e)) or np.linalg.norm(e) < 1e-6 for e in embs):
            raise SystemExit(f"zero/bad embedding at batch {i}. Stopped. Did patch_emb.py run?")
        store.collection.update(ids=res["ids"], embeddings=embs)
        if (i // bs) % 10 == 0:
            log(f"  {min(i + bs, len(ids))}/{len(ids)}")
    chk = store.collection.get(limit=400, include=["embeddings"])
    n = np.linalg.norm(np.asarray(chk["embeddings"], dtype=float), axis=1)
    log(f"RESULT done. stored vectors checked: {len(n)}  all-zero: {int((n < 1e-6).sum())}")

if __name__ == "__main__":
    sys.path.append('.'); sys.path.append('src')
    import transformers.utils.import_utils as iu; iu.check_torch_load_is_safe = lambda: None
    from rag.vectorstore import VectorStore
    from src.utils import load_beir_datasets, load_models
    m, c, t, g = load_models('contriever')
    corpus, q, r = load_beir_datasets('nfcorpus', 'test')
    s = VectorStore(m, t, g, corpus, 'cuda', 'nfcorpus_contriever_cosine', use_local=True)
    probe = np.asarray(s.get_embedding('prostate cancer and diet'), dtype=float).reshape(-1)
    print('RESULT query embedding norm', round(float(np.linalg.norm(probe)), 4))
    if np.linalg.norm(probe) < 1e-6:
        raise SystemExit("embedding function still returns zeros. Run patch_emb.py first.")
    reembed(s, log=lambda x: print(x, flush=True))
