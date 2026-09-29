from collections import defaultdict
from pathlib import Path
import hashlib
import re
import string
import unicodedata

def line_hash(line: str) -> bytes:
    return hashlib.sha256(line.encode("utf-8")).digest()


def exact_line_deduplication(
    input_files: list[Path],
    output_directory: Path,
):
    counts = Counter()

    for path in input_files:
        with open(path, "r") as f:
            for line in f:
                counts[line_hash(line)] += 1

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    for path in input_files:
        path = Path(path)
        output_path = output_directory / path.name

        with open(path, "r") as src, open(output_path, "w") as dst:
            for line in src:
                if counts[line_hash(line)] == 1:
                    dst.write(line)


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFD", text)

    text = "".join(
        ch for ch in text
        if not unicodedata.combining(ch)
    )

    text = text.lower()

    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    text = re.sub(r"\s+", " ", text).strip()

    return text


def make_ngrams(text: str, n: int) -> set[str]:
    words = normalize_text(text).split()

    if len(words) < n:
        return {" ".join(words)} if words else set()

    return {
        " ".join(words[i:i+n])
        for i in range(len(words) - n + 1)
    }


def seeded_hash(value: str, seed: int) -> int:
    data = f"{seed}:{value}".encode("utf-8")

    digest = hashlib.blake2b(
        data,
        digest_size=8,
    ).digest()

    return int.from_bytes(digest, "big")


def minhash_signature(
    ngram_set: set[str],
    num_hashes: int,
) -> list[int]:
    if not ngram_set:
        return [0] * num_hashes

    signature = []

    for seed in range(num_hashes):
        minimum = min(
            seeded_hash(ngram, seed)
            for ngram in ngram_set
        )
        signature.append(minimum)

    return signature


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0

    if not a or not b:
        return 0.0

    return len(a & b) / len(a | b)


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])

        return self.parent[x]

    def union(self, a: int, b: int):
        ra = self.find(a)
        rb = self.find(b)

        if ra != rb:
            self.parent[rb] = ra


def minhash_deduplication(
    input_files,
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory,
):
    input_files = [Path(p) for p in input_files]
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    input_files = sorted(input_files, key=lambda p: p.name)

    documents = [
        path.read_text()
        for path in input_files
    ]

    ngram_sets = [
        make_ngrams(text, ngrams)
        for text in documents
    ]

    signatures = [
        minhash_signature(s, num_hashes)
        for s in ngram_sets
    ]

    rows_per_band = num_hashes // num_bands

    buckets = defaultdict(list)

    for doc_id, signature in enumerate(signatures):
        for band_id in range(num_bands):
            start = band_id * rows_per_band
            end = start + rows_per_band

            band = tuple(signature[start:end])

            key = (band_id, band)

            buckets[key].append(doc_id)

    candidate_pairs = set()

    for docs in buckets.values():
        if len(docs) < 2:
            continue

        for i in range(len(docs)):
            for j in range(i + 1, len(docs)):
                candidate_pairs.add(
                    (docs[i], docs[j])
                )

    uf = UnionFind(len(documents))

    for i, j in candidate_pairs:
        similarity = jaccard(
            ngram_sets[i],
            ngram_sets[j],
        )

        if similarity >= jaccard_threshold:
            uf.union(i, j)

    clusters = defaultdict(list)

    for i in range(len(documents)):
        root = uf.find(i)
        clusters[root].append(i)

    keep = set()

    for members in clusters.values():
        keep.add(min(members))

    for i in sorted(keep):
        output_path = output_directory / input_files[i].name
        output_path.write_text(documents[i])
