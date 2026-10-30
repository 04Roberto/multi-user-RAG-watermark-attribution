"""
Builds validated_wmunit_pool.json — the piece missing from the other
group's repo. Their multi_user/assign_validated_pool_to_users.py expects
this file to already exist, containing only watermark tuples whose
generated text is KNOWN GOOD (passed verification), in this format:

    [
      {"wmunit": [e1, e2, relation], "valid_texts": ["...", "..."]},
      ...
    ]

We build it from OUR existing wmuint_doc.json — the output of
doc_llm_run_mutual() in src/main.py, which already ran on our fixed,
Llama7b-based pipeline ([INST] prompt formatting + OOM fixes applied).

Each entry in wmuint_doc.json has the shape:
    [wmunit, WT_list, WT_direct_flag_list, WT_simulate_flag_list]

where WT_direct_flag / WT_simulate_flag are 0=wrong, 1=correct, 2=unknown
per generated text (WT), matching stat_result()'s scoring convention.

"Valid" here means BOTH checkers said yes (flag == 1) — this is the same
"2 yes checker" filter that wm_inject_run()'s filter_wmunit() docstring in
our main.py claims to apply but currently doesn't (flagged with a NOTE
comment there). Building the pool this way effectively applies that
missing filter for the multi-user path, without touching the existing
single-user pipeline behaviour.
"""

import argparse
import json
import os


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path):
    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def build_validated_pool(wmuint_doc_list, require_both_checkers=True):
    pool = []
    skipped_no_valid_text = 0

    for entry in wmuint_doc_list:
        if len(entry) < 4:
            continue
        wmunit, wt_list, direct_flags, simulate_flags = entry[0], entry[1], entry[2], entry[3]
        
        # Skip self-referential tuples (e1 == e2) -- not a meaningful watermark fact
        if len(wmunit) >= 2 and wmunit[0] == wmunit[1]:
            skipped_no_valid_text += 1
            continue


        valid_texts = []
        for i, wt in enumerate(wt_list):
            d_ok = i < len(direct_flags) and direct_flags[i] == 1
            s_ok = i < len(simulate_flags) and simulate_flags[i] == 1

            if require_both_checkers:
                is_valid = d_ok and s_ok
            else:
                is_valid = d_ok or s_ok

            if is_valid:
                valid_texts.append(wt)

        if valid_texts:
            pool.append({"wmunit": wmunit, "valid_texts": valid_texts})
        else:
            skipped_no_valid_text += 1

    return pool, skipped_no_valid_text


def main():
    parser = argparse.ArgumentParser(
        description="Build validated_wmunit_pool.json from our fixed pipeline's wmuint_doc.json"
    )
    parser.add_argument(
        "--doc_path",
        default="output/wm_generate/nfcorpus/llama7b/10/wmuint_doc.json",
    )
    parser.add_argument(
        "--output_path",
        default="output/multi_user/nfcorpus/validated_pool/validated_wmunit_pool.json",
    )
    parser.add_argument(
        "--require_both_checkers",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no_require_both",
        dest="require_both_checkers",
        action="store_false",
    )
    args = parser.parse_args()

    wmuint_doc_list = load_json(args.doc_path)
    print(f"Loaded {len(wmuint_doc_list)} wmunit doc entries from {args.doc_path}")

    pool, skipped = build_validated_pool(wmuint_doc_list, args.require_both_checkers)

    print(f"Validated pool size: {len(pool)} wmunits with at least one valid text")
    print(f"Skipped (no valid text found): {skipped}")

    if len(pool) == 0:
        print(
            "\nWARNING: validated pool is empty. This means none of the generated "
            "watermark texts passed both checkers. Possible causes: not enough "
            "wmunits have been run through doc generation yet, or the checker "
            "criteria are too strict for the current model/data. Try "
            "--no_require_both to relax, or check wmuint_doc.json directly."
        )

    save_json(pool, args.output_path)
    print(f"\nSaved validated pool -> {args.output_path}")

    print("\nFirst 3 entries:")
    for item in pool[:3]:
        print(f"  wmunit: {item['wmunit']}")
        print(f"    valid_texts ({len(item['valid_texts'])}): {item['valid_texts'][0][:80]}...")


if __name__ == "__main__":
    main()
