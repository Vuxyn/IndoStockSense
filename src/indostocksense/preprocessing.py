"""Text preprocessing utilities for Indonesian stock news."""

import re


def clean_text(text: str) -> str:
    """Clean Indonesian news text before tokenization."""
    if text is None:
        return ""

    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

