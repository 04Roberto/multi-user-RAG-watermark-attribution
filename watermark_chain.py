"""
Phase A: Generate watermark tuples (per-user)

Implements A2, A3, A4 from the client's framework:
  A2 - Build User DB & fingerprint-code DB
  A3 - Create the watermark entity chain
  A4 - Generate the watermark relations
"""

import hmac
import hashlib
import random
from dataclasses import dataclass, field


def hmac_index(key: bytes, message: str, mod: int) -> int:
    digest = hmac.new(key, message.encode("utf-8"), hashlib.sha256).digest()
    as_int = int.from_bytes(digest, byteorder="big")
    return as_int % mod


@dataclass
class User:
    user_id: str
    user_key: bytes
    fingerprint: str = ""


def generate_user_key(master_key: bytes, user_id: str) -> bytes:
    return hmac.new(master_key, user_id.encode("utf-8"), hashlib.sha256).digest()


def generate_fingerprint(user_index: int, num_users: int, code_length: int = 64) -> str:
    rng = random.Random(f"{user_index}-{num_users}-fingerprint")
    return "".join(rng.choice("01") for _ in range(code_length))


def build_user_and_fingerprint_db(user_ids: list[str], master_key: bytes) -> list[User]:
    users = []
    for i, uid in enumerate(user_ids):
        user_key = generate_user_key(master_key, uid)
        fingerprint = generate_fingerprint(i, len(user_ids))
        users.append(User(user_id=uid, user_key=user_key, fingerprint=fingerprint))
    return users


def generate_watermark_entity_chain(
    user_key: bytes,
    entities: list[str],
    seed_index: int = 0,
    chain_length: int = 50,
) -> list[str]:
    if not entities:
        raise ValueError("entities list is empty")

    chain = [entities[seed_index % len(entities)]]
    for _ in range(chain_length - 1):
        prev_entity = chain[-1]
        idx = hmac_index(user_key, prev_entity, len(entities))
        chain.append(entities[idx])
    return chain


@dataclass
class WatermarkTuple:
    e1: str
    relation: str | None
    e2: str


def generate_watermark_relations(
    user_key: bytes,
    entity_chain: list[str],
    relations: list[str],
    p: float = 0.05,
) -> list[WatermarkTuple]:
    if not relations:
        raise ValueError("relations list is empty")

    tuples = []
    for e1, e2 in zip(entity_chain, entity_chain[1:]):
        pair_key = f"{e1}|{e2}"
        coin = hmac_index(user_key, f"coin:{pair_key}", 10_000) / 10_000.0

        if coin < p:
            r_idx = hmac_index(user_key, pair_key, len(relations))
            relation = relations[r_idx]
        else:
            relation = None

        tuples.append(WatermarkTuple(e1=e1, relation=relation, e2=e2))

    return tuples


def generate_user_watermark_tuples(
    user: User,
    entities: list[str],
    relations: list[str],
    chain_length: int = 50,
    seed_index: int = 0,
    p: float = 0.05,
) -> list[WatermarkTuple]:
    chain = generate_watermark_entity_chain(
        user.user_key, entities, seed_index=seed_index, chain_length=chain_length
    )
    return generate_watermark_relations(user.user_key, chain, relations, p=p)


def load_entities(path: str) -> list[str]:
    import json
    with open(path) as f:
        data = json.load(f)
    return list(data.keys())


def load_relations(path: str) -> list[str]:
    import json
    with open(path) as f:
        data = json.load(f)
    return list(data.keys())


def wm_tuples_to_wmunit_format(wm_tuples: list) -> list[list]:
    return [
        [t.e1, t.e2, t.relation]
        for t in wm_tuples
        if t.relation is not None
    ]


if __name__ == "__main__":
    import sys

    entities_path = sys.argv[1] if len(sys.argv) > 1 else "entities_dict_llm.json"
    relations_path = sys.argv[2] if len(sys.argv) > 2 else "relation_type_llm.json"

    entities = load_entities(entities_path)
    relations = load_relations(relations_path)
    print(f"Loaded {len(entities)} entities, {len(relations)} relations")

    master_key = b"REPLACE-WITH-A-REAL-SECRET-MASTER-KEY"
    users = build_user_and_fingerprint_db(["U001", "U002", "U003"], master_key)

    all_output = {}
    for user in users:
        print(f"\n=== {user.user_id} ===")
        print(f"user_key (hex): {user.user_key.hex()[:16]}...")

        wm_tuples = generate_user_watermark_tuples(
            user, entities, relations,
            chain_length=50,
            seed_index=0,
            p=0.05,
        )

        unique_entities_in_chain = len(set(t.e1 for t in wm_tuples) | {wm_tuples[-1].e2})
        real_relation_count = sum(1 for t in wm_tuples if t.relation is not None)
        print(f"Chain length: {len(wm_tuples) + 1} entities ({unique_entities_in_chain} unique)")
        print(f"Relations found: {real_relation_count}/{len(wm_tuples)} pairs")

        for t in wm_tuples:
            if t.relation is not None:
                print(f"  {t.e1} --{t.relation}--> {t.e2}")

        wmunit_style = wm_tuples_to_wmunit_format(wm_tuples)
        all_output[user.user_id] = wmunit_style

    import json
    for user_id, tuples in all_output.items():
        out_path = f"wmunit_{user_id}.json"
        with open(out_path, "w") as f:
            json.dump(tuples, f, indent=2)
        print(f"\nWrote {len(tuples)} watermark tuples for {user_id} -> {out_path}")
