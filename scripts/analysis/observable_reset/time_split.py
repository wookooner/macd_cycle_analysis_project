from __future__ import annotations

import argparse
import json

import pandas as pd

from lib import FEATURE_PATH, OUT_DIR, SPLIT_PATH, TEST_PATH, TRAIN_PATH, load_features


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-end", default="2023-12-31 23:59:59")
    parser.add_argument("--unlock-test", action="store_true", help="Write the locked test parquet for later OOS steps.")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_features(FEATURE_PATH).copy()
    df["entry_time"] = pd.to_datetime(df["entry_time"], errors="coerce")
    train_end = pd.Timestamp(args.train_end)
    train = df[df["entry_time"] <= train_end].copy()
    test = df[df["entry_time"] > train_end].copy()

    train.to_parquet(TRAIN_PATH, index=False)
    if args.unlock_test:
        test.to_parquet(TEST_PATH, index=False)
        test_written = True
    else:
        test_written = False

    payload = {
        "train_end": str(train_end),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "test_locked": not test_written,
        "train_path": str(TRAIN_PATH),
        "test_path": str(TEST_PATH) if test_written else "LOCKED_UNTIL_VALIDATE_OOS",
        "feature_path": str(FEATURE_PATH),
    }
    SPLIT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
