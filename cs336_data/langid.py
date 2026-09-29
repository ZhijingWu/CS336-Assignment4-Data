import fasttext


MODEL_PATH = "models/lid.176.bin"
model = fasttext.load_model(MODEL_PATH)


def identify_language(text: str) -> tuple[str, float]:
    text = " ".join(text.splitlines())
    
    labels, scores = model.predict(text, k=1)

    label = labels[0].replace("__label__", "")
    score = float(scores[0])

    return (label, score)