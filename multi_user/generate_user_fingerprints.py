import argparse
import json
import os


def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def generate_users(num_users, bit_length):
    max_users = 2 ** bit_length
    if num_users > max_users:
        raise ValueError(
            f"num_users={num_users} is too large for bit_length={bit_length}. "
            f"Maximum supported users is {max_users}."
        )

    user_db = []
    fingerprint_db = []

    for user_id in range(num_users):
        username = f"user_{user_id}"
        fingerprint_code = format(user_id, f"0{bit_length}b")

        user_db.append({
            "user_id": user_id,
            "username": username,
            "active": True
        })

        fingerprint_db.append({
            "user_id": user_id,
            "username": username,
            "fingerprint_code": fingerprint_code,
            "bit_length": bit_length
        })

    return user_db, fingerprint_db


def main():
    parser = argparse.ArgumentParser(description="Generate User DB and fingerprinting code DB for multi-user RAG-WM.")
    parser.add_argument("--num_users", type=int, default=128)
    parser.add_argument("--bit_length", type=int, default=8)
    parser.add_argument("--output_dir", type=str, default="output/multi_user/nfcorpus")
    args = parser.parse_args()

    user_db, fingerprint_db = generate_users(args.num_users, args.bit_length)

    user_db_path = os.path.join(args.output_dir, "user_db.json")
    fingerprint_db_path = os.path.join(args.output_dir, "fingerprint_db.json")

    save_json(user_db, user_db_path)
    save_json(fingerprint_db, fingerprint_db_path)

    print("Generated User DB and fingerprint DB")
    print(f"Users: {args.num_users}")
    print(f"Bit length: {args.bit_length}")
    print(f"User DB: {user_db_path}")
    print(f"Fingerprint DB: {fingerprint_db_path}")
    print("\nFirst 5 fingerprints:")
    for row in fingerprint_db[:5]:
        print(row)


if __name__ == "__main__":
    main()
