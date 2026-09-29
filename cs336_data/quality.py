from pathlib import Path

import fasttext

from cs336_data.train_quality import chunk_text


def gopher_quality_filter(text: str) -> bool:
    words = text.split()

    if len(words) < 50 or len(words) > 100000:
        return False

    avg_word_length = sum(len(word) for word in words) / len(words)
    if avg_word_length < 3 or avg_word_length > 10:
        return False

    lines = text.splitlines()
    if lines:
        ellipsis_ratio = sum(
            line.rstrip().endswith("...") for line in lines
        ) / len(lines)
        if ellipsis_ratio > 0.3:
            return False

    alpha_ratio = sum(
        any(ch.isalpha() for ch in word)
        for word in words
    ) / len(words)

    if alpha_ratio < 0.8:
        return False

    return True


ROOT = Path(__file__).resolve().parents[1]
QUALITY_MODEL_PATH = ROOT / "models/quality_classifier.bin"

quality_model = fasttext.load_model(str(QUALITY_MODEL_PATH))


def classify_quality(text: str) -> tuple[str, float]:
    chunks = chunk_text(text)

    if not chunks:
        chunks = [" ".join(text.splitlines())]

    scores_by_label = {
        "wiki": 0.0,
        "cc": 0.0,
    }

    for chunk in chunks:
        labels, scores = quality_model.predict(chunk, k=2)

        for label, score in zip(labels, scores):
            label = label.replace("__label__", "")
            scores_by_label[label] += float(score)

    n = len(chunks)

    wiki_score = scores_by_label["wiki"] / n
    cc_score = scores_by_label["cc"] / n

    if wiki_score > cc_score:
        return "wiki", wiki_score

    return "cc", cc_score