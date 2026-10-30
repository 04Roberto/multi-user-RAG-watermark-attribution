import argparse
import hashlib
import hmac
import json
import os


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def hmac_score(user_key, item):
    msg = json.dumps(item["wmunit"], sort_keys=True)
    digest = hmac.new(
        user_key.encode("utf-8"),
        msg.encode("utf-8"),
        hashlib.sha256
    ).hexdigest()
    return int(digest, 16)


def main():
    parser = argparse.ArgumentParser(description="Assign validated RAG-WM tuples to users.")
    parser.add_argument("--fingerprint_db", default="output/multi_user/nfcorpus/fingerprint_db.json")
    parser.add_argument("--validated_pool", default="output/multi_user/nfcorpus/validated_pool/validated_wmunit_pool.json")
    parser.add_argument("--output_dir", default="output/multi_user/nfcorpus/validated_user_assignments")
    parser.add_argument("--num_users", type=int, default=5)
    parser.add_argument("--tuples_per_user", type=int, default=30)
    args = parser.parse_args()

    fingerprint_db = load_json(args.fingerprint_db)[:args.num_users]
    pool = load_json(args.validated_pool)

    needed = args.num_users * args.tuples_per_user
    if needed > len(pool):
        raise ValueError(
            f"Not enough validated tuples for non-overlap assignment. "
            f"Need {needed}, but pool only has {len(pool)}."
        )

    assigned_global = set()
    all_assignments = []

    for user in fingerprint_db:
        user_key = user["fingerprint_code"]

        ranked_pool = sorted(
            pool,
            key=lambda item: hmac_score(user_key, item)
        )

        selected = []
        for item in ranked_pool:
            tuple_key = tuple(item["wmunit"])
            if tuple_key in assigned_global:
                continue

            selected.append(item)
            assigned_global.add(tuple_key)

            if len(selected) == args.tuples_per_user:
                break

        if len(selected) < args.tuples_per_user:
            raise ValueError(f"Could not assign enough tuples for user {user['user_id']}.")

        assignment = {
            "user_id": user["user_id"],
            "username": user["username"],
            "fingerprint_code": user["fingerprint_code"],
            "tuple_count": len(selected),
            "assigned_validated_tuples": selected
        }

        user_path = os.path.join(args.output_dir, f"user_{user['user_id']}_validated_assignment.json")
        save_json(assignment, user_path)
        all_assignments.append(assignment)

        print(f"User {user['user_id']} assigned {len(selected)} validated tuples")
        print("First 3:")
        for item in selected[:3]:
            print(" ", item["wmunit"])

    combined_path = os.path.join(args.output_dir, "all_validated_assignments.json")
    save_json(all_assignments, combined_path)

    print("\nCombined assignment saved:", combined_path)
    print("Total assigned tuples:", len(assigned_global))


if __name__ == "__main__":
    main()
