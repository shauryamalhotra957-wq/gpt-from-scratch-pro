"""Download Tiny Shakespeare for a larger GPT-from-scratch experiment."""

from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve


URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def main() -> None:
    target = Path(__file__).resolve().parents[1] / "data" / "tiny_shakespeare.txt"
    target.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(URL, target)
    print(f"Downloaded {URL}")
    print(f"Wrote {target}")


if __name__ == "__main__":
    main()

