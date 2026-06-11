"""Convert a # %% tagged Python script to a Jupyter Notebook (.ipynb)."""
import sys
import json
import re
from pathlib import Path

def convert(py_path: str):
    src = Path(py_path).read_text(encoding="utf-8")
    # Split on cell markers
    parts = re.split(r"^# %%.*$", src, flags=re.MULTILINE)
    
    cells = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        # Detect markdown cells (every line starts with # or is blank)
        lines = part.splitlines()
        is_markdown = all(l.startswith("#") or l == "" for l in lines)
        if is_markdown:
            # Strip leading "# " from each line
            md_lines = [re.sub(r"^# ?", "", l) for l in lines]
            cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": [l + "\n" for l in md_lines]
            })
        else:
            cells.append({
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [l + "\n" for l in lines]
            })

    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {"name": "python", "version": "3.10.0"}
        },
        "cells": cells
    }

    out_path = Path(py_path).with_suffix(".ipynb")
    out_path.write_text(json.dumps(nb, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Converted: {py_path} -> {out_path}")

if __name__ == "__main__":
    convert(sys.argv[1])
