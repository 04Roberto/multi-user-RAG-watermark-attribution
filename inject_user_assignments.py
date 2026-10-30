"""
Injects each simulated user's assigned validated watermark tuples into the
vector DB, so they're actually retrievable.

FIX: the original Visiter.inject_wm() only ever requests the single top-1
most relevant document for each tuple (n_results = len(str_lines) = 1), with
no awareness of what other tuples/users already landed on. This caused every
one of our 9 test injections (3 users x 3 tuples) to collide into the exact
SAME document, because each injection made that document even more relevant
to the next related query -- a feedback loop. Confirmed via real output:
all 9 injections landed on id_3611.

This script does NOT call visiter.inject_wm() directly. Instead it
replicates the same underlying steps (search_context + update_context) but
tracks a shared `used_doc_ids` set across the WHOLE run (all users, all
tuples), requests the top 10 candidates instead of just 1, and picks the
first candidate not already used. Falls back to top-1 only if every
candidate in the top 10 is already taken (prints a warning in that case).
"""

import transformers.utils.import_utils as import_utils
import_utils.check_torch_load_is_safe = lambda: None

import os
import sys
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'src'))

import argparse
import glob
import json
import torch
import numpy as np
import random

from watermark_role import Advisor, Visiter, Checker
from rag.vectorstore import VectorStore, check_collection
from src.utils import load_json, save_json, load_beir_datasets, load_models, Log, file_exist
from src.models import create_model

seed = 633
torch.backends.cudnn.deterministic = True
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def inject_wm_excluding(visiter, wm_unit, text, used_ids, top_k=10):
    """
    Like Visiter.inject_wm(), but avoids colliding with documents already
    used elsewhere in this run.
    """
    user_question = f"What is the relationship between {wm_unit[0]} and {wm_unit[1]} ?"
    context_response = visiter.vectorstore.search_context(user_question, top_k)
    candidate_ids = context_response['ids'][0]

    chosen_id = None
    for cid in candidate_ids:
        if cid not in used_ids:
            chosen_id = cid
            break

    if chosen_id is None:
        chosen_id = candidate_ids[0]
        print(f"  WARNING: all top-{top_k} candidates already used for {wm_unit}, "
              f"falling back to top-1 ({chosen_id}) -- may collide with another user's watermark")

    visiter.vectorstore.update_context(chosen_id, text)
    used_ids.add(chosen_id)
    return chosen_id


def wm_inject_run_multiuser(visiter, assignment_dir, output_dir, texts_per_tuple=1, logger=None):
    assignment_files = sorted(glob.glob(os.path.join(assignment_dir, "user_*_validated_assignment.json")))

    if not assignment_files:
        print(f"No user_*_validated_assignment.json files found in {assignment_dir}")
        return

    # Shared across ALL users and ALL tuples in this run -- this is the fix.
    used_doc_ids = set()

    for af in assignment_files:
        assignment = load_json(af)
        user_id = assignment["user_id"]
        username = assignment.get("username", f"user_{user_id}")

        injected = []
        for item in assignment["assigned_validated_tuples"]:
            wmunit = item["wmunit"]
            valid_texts = item["valid_texts"][:texts_per_tuple]

            if not valid_texts:
                continue

            loc_list = []
            try:
                for text in valid_texts:
                    chosen_id = inject_wm_excluding(visiter, wmunit, text, used_doc_ids, top_k=50)
                    loc_list.append(chosen_id)

                injected.append([wmunit, loc_list, valid_texts])
                print(f"  [{username}] injected {wmunit} -> loc_list: {loc_list}")
            except Exception as e:
                print(f"  [{username}] FAILED to inject {wmunit}: {e}")
                if logger:
                    logger.error(f"Injection failed for {username}, wmunit {wmunit}: {e}")
                continue

        out_path = os.path.join(output_dir, f"user_{user_id}_injected.json")
        save_json(injected, out_path)
        print(f"User {user_id} ({username}): injected {len(injected)}/{len(assignment['assigned_validated_tuples'])} wmunits -> {out_path}")

    print(f"\nTotal distinct documents used across all users: {len(used_doc_ids)}")


def parse_args():
    parser = argparse.ArgumentParser(description="Inject per-user validated watermark assignments into the vector DB")

    parser.add_argument("--eval_model_code", type=str, default="contriever")
    parser.add_argument("--eval_dataset", type=str, default="nfcorpus")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--score_function", type=str, default="cosine")
    parser.add_argument("--top_k", type=int, default=5)

    parser.add_argument("--model_config_path", default=None, type=str)
    parser.add_argument("--model_name_rllm", default="llama7b", type=str)
    parser.add_argument("--model_name_llm", type=str, default="llama7b")

    parser.add_argument("--gpu_id", type=int, default=0)

    parser.add_argument("--basepath", type=str,
                         default="/fred/oz479/ICT30017-P39-RAG-Watermarking-for-ownership-user-atribution/ragwm")

    parser.add_argument("--assignment_dir", type=str,
                         default="output/multi_user/nfcorpus/validated_user_assignments")
    parser.add_argument("--output_dir", type=str,
                         default="output/multi_user/nfcorpus/validated_user_assignments")
    parser.add_argument("--texts_per_tuple", type=int, default=1)
    parser.add_argument("--clean_db", type=int, default=0)

    args = parser.parse_args()
    print(args)
    return args


if __name__ == "__main__":
    args = parse_args()
    torch.cuda.set_device(args.gpu_id)
    device = "cuda"

    basepath = os.path.join(args.basepath, "output", "wm_generate", args.eval_dataset)
    LOG_FILE = os.path.join(basepath, args.model_name_rllm, "multiuser_inject_log.log")
    file_exist(LOG_FILE)
    logger = Log(log_file=LOG_FILE).get(__file__)

    args.model_config_path = f"model_configs/{args.model_name_llm}_config.json"
    llm = create_model(args.model_config_path)

    args.model_config_path = f"model_configs/{args.model_name_rllm}_config.json"
    rllm = create_model(args.model_config_path)

    advisor = Advisor(llm)
    checker = Checker(llm)

    collection_name = args.eval_dataset + "_" + args.eval_model_code + "_" + args.score_function
    print(collection_name)

    model, c_model, tokenizer, get_emb = load_models(args.eval_model_code)

    if args.eval_dataset == "msmarco":
        corpus, queries, qrels = load_beir_datasets("msmarco", "train")
    else:
        corpus, queries, qrels = load_beir_datasets(args.eval_dataset, args.split)

    collection_exist, collection_len = check_collection(collection_name)
    if not (collection_exist and collection_len > 0):
        print(f"please run rag/vectorstore.py to create vectorstore for {collection_name}")
        sys.exit(1)

    vectorstore = VectorStore(model, tokenizer, get_emb, corpus, device, collection_name, use_local=True)

    if args.clean_db == 1:
        print("Wiping vector DB collection before injecting (--clean_db 1)")
        vectorstore.clean_collect()

    visiter = Visiter(llm, rllm, vectorstore)

    print(f"\nInjecting user assignments from {args.assignment_dir}\n")
    wm_inject_run_multiuser(
        visiter,
        assignment_dir=args.assignment_dir,
        output_dir=args.output_dir,
        texts_per_tuple=args.texts_per_tuple,
        logger=logger,
    )
