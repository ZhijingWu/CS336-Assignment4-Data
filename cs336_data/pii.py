import re


def mask_emails(text: str) -> tuple[str, int]:
    pattern = r"[^@\s]+@[^@\s]+\.[^@\s]+"
    return re.subn(pattern, "|||EMAIL_ADDRESS|||", text)


def mask_phone_numbers(text: str) -> tuple[str, int]:
    pattern = r"(?:\(\d{3}\)|\d{3})[- ]?\d{3}[- ]?\d{4}"
    return re.subn(pattern, "|||PHONE_NUMBER|||", text)


def mask_ips(text: str) -> tuple[str, int]:
    pattern = r"(?:25[0-5]|2[0-4]\d|1?\d?\d)\.(?:25[0-5]|2[0-4]\d|1?\d?\d)\.(?:25[0-5]|2[0-4]\d|1?\d?\d)\.(?:25[0-5]|2[0-4]\d|1?\d?\d)"
    return re.subn(pattern, "|||IP_ADDRESS|||", text)