"""
user_greenlist.py - per-user "green word" watermark on generated text (green-list idea as in WARD / KGW).
Standalone module. Put it in src/ on the HPC. Detection and the selftest need only numpy (no torch / no GPU).

  python user_greenlist.py selftest                      # checks the maths, no model needed
  python user_greenlist.py hpc --model_path <HF path>    # real Llama run on the 3 test users
"""
import argparse, hashlib, hmac, math, sys
import numpy as np

GAMMA = 0.25   # fraction of the vocabulary that is "green" for a user
DELTA = 5.0    # how strongly we nudge towards green words


def make_user_key(user_id):
    """Stand-in user key for the 3 test users. Swap for the real key from user_db later."""
    return hashlib.sha256(f"user_key_{user_id}".encode()).digest()


def green_key(user_key):
    """Separate sub-key for the green words, so it is independent of the entity-chain key."""
    return hmac.new(user_key, b"greenlist", hashlib.sha256).digest()


class GreenLists:
    """Green word set for one key, depends on the previous token. Cached."""
    def __init__(self, key, vocab_size, gamma=GAMMA):
        self.key, self.V, self.gamma = key, vocab_size, gamma
        self.idx, self.mask = {}, {}

    def _build(self, prev):
        seed = int.from_bytes(hmac.new(self.key, str(int(prev)).encode(), hashlib.sha256).digest()[:8], "big")
        idx = np.random.default_rng(seed).permutation(self.V)[: int(self.gamma * self.V)]
        m = np.zeros(self.V, dtype=bool); m[idx] = True
        self.idx[prev], self.mask[prev] = idx, m

    def green_ids(self, prev):
        if prev not in self.idx: self._build(prev)
        return self.idx[prev]

    def is_green(self, prev, tok):
        if prev not in self.mask: self._build(prev)
        return bool(self.mask[prev][tok])


class UserGreenListProcessor:
    """Plug into model.generate(logits_processor=LogitsProcessorList([...])). Needs torch (already on the HPC)."""
    def __init__(self, key, vocab_size, gamma=GAMMA, delta=DELTA):
        self.gl, self.delta = GreenLists(key, vocab_size, gamma), delta

    def __call__(self, input_ids, scores):
        import torch
        for b in range(scores.shape[0]):
            ids = torch.as_tensor(self.gl.green_ids(input_ids[b, -1].item()), device=scores.device, dtype=torch.long)
            scores[b, ids] += self.delta
        return scores


def detect(token_ids, key, vocab_size, gamma=GAMMA):
    """token_ids: a list of ints. Each token is checked against the green list made from the token before it.
    Each (previous token, token) pair is counted only once, so repeated phrases don't skew the score.
    Returns (z, green_count, pairs_checked)."""
    gl = GreenLists(key, vocab_size, gamma)
    pairs = set(zip(token_ids[:-1], token_ids[1:]))
    total = len(pairs)
    if total < 1:
        return 0.0, 0, 0
    green = sum(gl.is_green(p, t) for p, t in pairs)
    z = (green - gamma * total) / math.sqrt(total * gamma * (1 - gamma))
    return z, int(green), total


def selftest(V=32000, n_tokens=200, users=(1, 2, 3)):
    print("Self test (fake model, random logits). Each text should score highest on its own user.")
    rng = np.random.default_rng(0)
    keys = {u: green_key(make_user_key(u)) for u in users}

    def gen(u):
        proc = GreenLists(keys[u], V) if u else None
        ids = [5]
        for _ in range(n_tokens):
            l = rng.normal(size=V) * 2.0
            if proc is not None: l[proc.green_ids(ids[-1])] += DELTA
            p = np.exp(l - l.max()); p /= p.sum()
            ids.append(int(rng.choice(V, p=p)))
        return ids

    print("text from \\ key tested " + "".join(f"{'user'+str(u):>9}" for u in users))
    ok = True
    for u in list(users) + [0]:
        ids = gen(u)
        zs = {v: detect(ids, keys[v], V)[0] for v in users}
        print(f"{('user'+str(u)) if u else 'no watermark':<23}" + "".join(f"{zs[v]:9.2f}" for v in users))
        if u: ok &= (max(zs, key=zs.get) == u and zs[u] > 4)
        else: ok &= all(abs(z) < 4 for z in zs.values())
    print("RESULT:", "PASS" if ok else "FAIL")


def hpc(model_path, users, max_new_tokens):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")
    V = model.get_output_embeddings().weight.shape[0]
    prompts = ["What is the relationship between vitamin D and bone health?",
               "Explain how diet can affect heart disease.",
               "What do we know about omega-3 and inflammation?"]
    keys = {u: green_key(make_user_key(u)) for u in users}
    print("answer for \\ key tested " + "".join(f"{'user'+str(u):>9}" for u in users) + "   (average z over prompts)")
    for u in users:
        zsum = {v: 0.0 for v in users}
        for p in prompts:
            enc = tok(f"[INST] {p} [/INST]", return_tensors="pt").to(model.device)
            proc = UserGreenListProcessor(keys[u], V)
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7,
                                     logits_processor=LogitsProcessorList([proc]))
            ids = [enc.input_ids[0, -1].item()] + out[0, enc.input_ids.shape[1]:].tolist()
            for v in users:
                zsum[v] += detect(ids, keys[v], V)[0] / len(prompts)
            torch.cuda.empty_cache()
        print(f"user{u:<19}" + "".join(f"{zsum[v]:9.2f}" for v in users))
    print("Good result: the biggest number in each row is on the diagonal (own user), clearly above ~4.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["selftest", "hpc"])
    ap.add_argument("--model_path"); ap.add_argument("--max_new_tokens", type=int, default=200)
    ap.add_argument("--users", type=int, nargs="+", default=[1, 2, 3])
    a = ap.parse_args()
    if a.mode == "selftest": selftest()
    else:
        if not a.model_path: sys.exit("need --model_path")
        hpc(a.model_path, a.users, a.max_new_tokens)
