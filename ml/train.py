"""Entry point compatível com ``python -m ml.train``."""

from .cli import main


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(["train", *__import__("sys").argv[1:]]))
