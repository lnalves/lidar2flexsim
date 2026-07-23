"""Entry point compatível com ``python -m ml.infer``."""

from .cli import main


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(["infer", *__import__("sys").argv[1:]]))
