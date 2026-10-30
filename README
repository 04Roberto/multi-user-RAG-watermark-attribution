# Multi-user RAG watermarking for ownership and user attribution

ICT30017 / ICT30018, project UG_ICT_P39. This repo is a fork of [wanghao512/ragwm](https://github.com/wanghao512/ragwm) (RAG-WM). We kept the original pipeline and added a multi-user layer so a leak can be traced back to one user.

## What we added

| File | What it does |
|---|---|
| `watermark_chain.py` | Prototype of the client spec (steps A2-A4): user keys, HMAC entity chain, relations. The chain hashes the step number into each step, because the plain `HMAC(user_key, entity)` chain loops back on itself. |
| `build_validated_pool.py` | Keeps only watermark tuples whose generated text passed both checks (direct and simulate). Skips self-referencing tuples. |
| `multi_user/generate_user_fingerprints.py` | Builds the user DB and fingerprint DB. **The fingerprints are placeholder binary counters, not real BKM10 codes.** Adapted from the other group's code. |
| `multi_user/assign_validated_pool_to_users.py` | Gives each user their own tuples with no overlap. Adapted from the other group's code. |
| `inject_user_assignments.py` | Injects each user's watermark documents into the vector DB. Tracks used documents and searches the top 50, so every tuple lands in a different document. |
| `src/user_greenlist.py` | Per-user "green word" watermark on generated text (green-list idea, as in WARD). Needs a user key. |
| `trace_leak.py` | Scores a leaked text for every user (green-word z-score plus how many of the user's facts appear). |
| `gen_leak.py` | Makes a watermarked sample answer for one user, used to test `trace_leak.py`. |

Fixes to the original code: `[INST]...[/INST]` prompt format and yes/no matching in `src/main.py`, the missing driver loop, a stats index bug, an out-of-memory fix and an output cap in `src/models/Llama.py`, an `--aim_count` argument, and an optional per-user key in `Llama.query()` (`set_user_key`). Changes are marked with `FIX:` or `NOTE:` comments.

## Setup

- Python environment with the packages in `requirement.txt`. We ran on the Swinburne HPC with one GPU node.
- Not in this repo (too large): the Llama-7B-chat weights, the BEIR datasets (`nfcorpus`) and generated outputs. Download the datasets with BEIR and put the weights in `weights/llama-7b-chat`.
- `model_configs/llama7b_config.json` contains our HPC weights path. Change `name` to your own path.

## How to run

Original pipeline (see the original README for details):

1. `python entity_generate/generate_entity_llm_check.py --eval_dataset 'nfcorpus' --dataset_prob 1`
2. `python entity_generate/generate_hash_entity.py -t scratch --eval_dataset "nfcorpus" --entity_num 100 --edge_prob 0.05`
3. `src/main.py` with `--doc 1`, then `--inject 1`, then `--verify 1 --stat 1` (use `--aim_count 26`, `--model_name_llm llama7b`, `--model_config_path model_configs/llama7b_config.json`).

Multi-user layer (use `--help` on each script for its arguments):

1. `python build_validated_pool.py`
2. `python multi_user/generate_user_fingerprints.py`
3. `python multi_user/assign_validated_pool_to_users.py`
4. `python inject_user_assignments.py` (use `--clean_db 1` for a fresh database)
5. Green-word test: `python src/user_greenlist.py selftest`, then `python src/user_greenlist.py hpc --model_path <weights folder>`
6. Tracing: make a sample with `python gen_leak.py <weights folder> <user id> leak.txt`, then `python trace_leak.py --text_file leak.txt --model_path <weights folder> --users 0 1 2`

Set `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` on the HPC, because it has no internet.

## Results (3 test users: ids 0, 1, 2)

- Each user got 3 watermark facts, 9 in total, injected into 9 different documents with no overlap.
- Green-word watermark (nudge strength 5, 300 tokens, average over 3 questions): each user's own text scored highest on their own key (z = 6.53, 4.31, 4.31). The wrong keys scored between about -0.6 and 1.6. A first test with strength 2 failed.
- Tracing one 300-token answer made for user 2: user 2 ranked first (z = 3.06), ahead of user 0 (-0.63) and user 1 (1.14). It was below our fixed threshold of z > 4, so no user was accused. Longer text should raise the score, but we did not test this.

## Limitations

- Fingerprints are placeholders, not real BKM10 codes.
- Detection and tracing as in the client spec (binomial test, accused set) is only approximated by `trace_leak.py`.
- Single-user verification accuracy dropped from 92% to 42% at one point, and we did not find the cause.
- Some stored watermark texts contain leftover prompt wording (for example "Your refined text"). The validation check let them through.
- The green-word test ran at temperature 0.7. The RAG pipeline runs near temperature 0.01, which we did not test. Paraphrasing attacks were not tested.
- Poisoning and other attacks are described in our threat model framework but were not run.
- Only 3 test users were used.

## Credits

Based on RAG-WM by wanghao512. The fingerprint and assignment scripts were adapted from another student group's multi-user implementation. The green-word idea follows WARD (Jovanovic, Staab and Vechev, ICLR 2025) and the multi-user watermark paper by Cohen, Hoover and Schoenbach (IEEE S&P 2025).

## Testing, fixes and known issues (final sprint)

### Embedding fault found and fixed
`contriever_get_emb` in `src/utils.py` returned a vector of zeros whenever the vector store called it
(it expected a string but the store passes tokenised tensors). Every stored and queried vector was zero,
so retrieval was comparing ties. The function now handles the tokenised input.
After pulling this fix, rebuild the vectors and re-inject the watermark facts:

    python reembed_all.py
    python inject_user_assignments.py --clean_db 1

Retrieval-based results measured before this fix should not be trusted. This includes the earlier
92% to 42% verification result, which has not been rechecked.

Also note: `VectorStore.update_context(id, text)` appends text to a document, it does not replace it.

### Knowledge-base poisoning test (`poison_test.py`)
Damages 0/25/50/75/100% of the 9 watermark documents (3 users x 3 facts) in three ways and counts how many
facts still verify (document in the top 5 for the injection query and at least 60% of the watermark words
still present). Documents are restored after every trial.

    python poison_test.py --top_k 5 --trials 5

Results (facts still verified, out of 9, average of 5 trials, baseline 9/9):

| Documents poisoned | Overwrite | Add unrelated text | Delete half the words |
|---|---|---|---|
| 0% | 9.0 | 9.0 | 9.0 |
| 25% | 7.0 | 7.0 | 8.0 |
| 50% | 5.0 | 5.4 | 7.2 |
| 75% | 2.0 | 3.0 | 6.4 |
| 100% | 0.0 | 1.2 | 6.0 |

This tests edits to stored text only. It is not the embedding-perturbation attack from the threat model.

### Embedding perturbation script (`embed_attack.py`, not validated)
Nudges the stored vector of each watermark document (directed away from its query, or random as a control)
with increasing size and records how many facts still verify. It was tested on a mock store only and has
not been run on the corrected database, so there are no results to report.

### Not finished
Real fingerprints and the full binomial-test detection from the multi-user specification, the gradient-based
embedding attack, and a recheck of verification accuracy at full scale.
