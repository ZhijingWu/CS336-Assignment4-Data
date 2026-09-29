from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def simple_tokenize(text: str) -> list[int]:
    """
    Smoke-test tokenizer only.

    UTF-8 bytes are mapped to positive integer IDs.
    This is NOT the real GPT-2 tokenizer.
    """
    return [int(b) + 1 for b in text.encode("utf-8")]


EOS_TOKEN_ID = 256


def tokenize_jsonl_directory(
    input_dir: Path,
    output_path: Path,
):
    all_ids = []

    jsonl_files = sorted(
        Path(input_dir).glob("*.jsonl")
    )

    for path in jsonl_files:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)

                text = obj["text"]

                ids = simple_tokenize(text)
                ids.append(EOS_TOKEN_ID)

                all_ids.extend(ids)

    ids_array = np.array(
        all_ids,
        dtype=np.uint16,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    ids_array.tofile(output_path)

    print(
        f"Smoke-test tokenized {len(all_ids):,} tokens "
        f"into {output_path}"
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-path",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    tokenize_jsonl_directory(
        input_dir=args.input_dir,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()