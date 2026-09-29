import fasttext


NSFW_MODEL_PATH = "models/jigsaw_fasttext_bigrams_nsfw_final.bin"
TOXIC_MODEL_PATH = "models/jigsaw_fasttext_bigrams_hatespeech_final.bin"

nsfw_model = fasttext.load_model(NSFW_MODEL_PATH)
toxic_model = fasttext.load_model(TOXIC_MODEL_PATH)


def classify_nsfw(text: str) -> tuple[str, float]:
    text = " ".join(text.splitlines())

    labels, scores = nsfw_model.predict(text, k=1)

    label = labels[0].replace("__label__", "")
    score = float(scores[0])

    return label, score


def classify_toxic_speech(text: str) -> tuple[str, float]:
    text = " ".join(text.splitlines())

    labels, scores = toxic_model.predict(text, k=1)

    label = labels[0].replace("__label__", "")
    score = float(scores[0])

    return label, score