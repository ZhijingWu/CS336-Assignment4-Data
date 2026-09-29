from pathlib import Path

import fasttext


ROOT = Path(__file__).resolve().parents[1]

WIKI_PATH = ROOT / "tests/fixtures/high_quality_wiki_reference.txt"
CC_PATH = ROOT / "tests/fixtures/low_quality_cc.txt"

TRAIN_PATH = ROOT / "data/quality_train.txt"
MODEL_PATH = ROOT / "models/quality_classifier.bin"


def chunk_text(text: str, chunk_size: int = 200) -> list[str]:
    words = text.split()

    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = words[i : i + chunk_size]

        if len(chunk) >= 20:
            chunks.append(" ".join(chunk))

    return chunks

def main():
    wiki_text = WIKI_PATH.read_text()
    cc_text = CC_PATH.read_text()

    wiki_chunks = chunk_text(wiki_text, chunk_size=50)
    cc_chunks = chunk_text(cc_text, chunk_size=50)

    n = min(len(wiki_chunks), len(cc_chunks))

    wiki_chunks = wiki_chunks[:n]
    cc_chunks = cc_chunks[:n]

    print(f"balanced wiki chunks: {len(wiki_chunks)}")
    print(f"balanced cc chunks: {len(cc_chunks)}")

    TRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)

    with TRAIN_PATH.open("w") as f:
        for text in wiki_chunks:
            f.write(f"__label__wiki {text}\n")

        for text in cc_chunks:
            f.write(f"__label__cc {text}\n")

    model = fasttext.train_supervised(
        input=str(TRAIN_PATH),
        epoch=25,
        lr=0.5,
        wordNgrams=2,
        dim=100,
    )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(MODEL_PATH))

    print(f"saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
