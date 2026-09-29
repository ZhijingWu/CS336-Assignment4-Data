from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
from collections import Counter
from pathlib import Path

from fastwarc.warc import ArchiveIterator
from xopen import xopen

from cs336_data.harmful import classify_nsfw, classify_toxic_speech
from cs336_data.pii import mask_emails, mask_phone_numbers, mask_ips
from cs336_data.quality import gopher_quality_filter, classify_quality


def filter_document(text: str) -> tuple[str | None, str]:
    """
    Returns:
        (filtered_text, reason)

    If filtered_text is None, the document was discarded.
    """

    nsfw_label, _ = classify_nsfw(text)
    if nsfw_label == "nsfw":
        return None, "nsfw"

    toxic_label, _ = classify_toxic_speech(text)
    if toxic_label == "toxic":
        return None, "toxic"

    if not gopher_quality_filter(text):
        return None, "gopher"

    quality_label, _ = classify_quality(text)
    if quality_label != "wiki":
        return None, "quality"

    text, email_count = mask_emails(text)
    text, phone_count = mask_phone_numbers(text)
    text, ip_count = mask_ips(text)

    modified = email_count + phone_count + ip_count

    if modified > 0:
        return text, "kept_modified"

    return text, "kept"


def process_single_wet_file(
    input_path: str,
    output_path: str,
) -> dict[str, int]:
    stats = Counter()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with xopen(input_path, "rb") as f_in, output_path.open(
        "w",
        encoding="utf-8",
    ) as f_out:
        for record in ArchiveIterator(f_in):
            try:
                raw = record.reader.read()
                text = raw.decode("utf-8", errors="ignore")
            except Exception:
                stats["decode_error"] += 1
                continue

            stats["total"] += 1

            filtered_text, reason = filter_document(text)
            stats[reason] += 1

            if filtered_text is None:
                continue

            # One document per JSON line.
            f_out.write(
                json.dumps(
                    {"text": filtered_text},
                    ensure_ascii=False,
                )
            )
            f_out.write("\n")

    return dict(stats)


def merge_stats(
    total_stats: Counter,
    new_stats: dict[str, int],
) -> None:
    for key, value in new_stats.items():
        total_stats[key] += value


def run_parallel_filtering(
    input_dir: Path,
    output_dir: Path,
    workers: int | None = None,
) -> Counter:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    wet_files = sorted(input_dir.glob("*.warc.wet.gz"))

    if not wet_files:
        raise FileNotFoundError(
            f"No *.warc.wet.gz files found under {input_dir}"
        )

    if workers is None:
        try:
            workers = len(os.sched_getaffinity(0))
        except AttributeError:
            workers = os.cpu_count() or 1

    total_stats = Counter()

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers
    ) as executor:
        futures = []

        for wet_path in wet_files:
            output_path = (
                output_dir
                / f"{wet_path.name}.jsonl"
            )

            future = executor.submit(
                process_single_wet_file,
                str(wet_path),
                str(output_path),
            )
            futures.append(future)

        for future in concurrent.futures.as_completed(futures):
            stats = future.result()
            merge_stats(total_stats, stats)

    return total_stats


def print_stats(stats: Counter) -> None:
    total = stats.get("total", 0)

    print("\n=== Filtering statistics ===")
    print(f"total: {total}")

    for key in [
        "nsfw",
        "toxic",
        "gopher",
        "quality",
        "kept_modified",
        "kept",
        "decode_error",
    ]:
        value = stats.get(key, 0)

        if total > 0:
            ratio = value / total
        else:
            ratio = 0.0

        print(
            f"{key:15s}: "
            f"{value:8d} "
            f"({ratio:.2%})"
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    stats = run_parallel_filtering(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        workers=args.workers,
    )

    print_stats(stats)


if __name__ == "__main__":
    main()