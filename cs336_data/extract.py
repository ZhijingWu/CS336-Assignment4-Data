from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.encoding import detect_encoding


def extract_text_from_html_bytes(html_bytes: bytes) -> str:
    try:
        bytes_to_text = html_bytes.decode("utf-8")
    except UnicodeDecodeError:
        code_kind = detect_encoding(html_bytes)
        bytes_to_text = html_bytes.decode(code_kind)

    text_str = extract_plain_text(bytes_to_text)
    return text_str