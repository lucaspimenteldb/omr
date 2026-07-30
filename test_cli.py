#!/usr/bin/env python3
"""
Testa o OMR a partir de arquivos de imagem, sem precisar subir a API.

Uso:
    python test_cli.py samples/gabarito_matematica.png
    python test_cli.py foto1.jpg foto2.jpg --debug
    python test_cli.py test_images/*.png --json

Opções:
    --debug   salva em debug/<nome>_debug.png uma imagem anotada (bolhas + status)
    --json    imprime o resultado bruto em JSON (útil para conferência/scripts)
"""
import argparse
import json
import os
import sys

import cv2

from omr import engine, OMRError

STATUS_LABEL = {
    "OK": "OK      ",
    "BLANK": "EM BRANCO",
    "MULTIPLE": "MULTIPLA ",
    "REVIEW": "REVISAR  ",
}


def print_report(name, result):
    s = result["summary"]
    print(f"\n=== {name} ===")
    print(f"{result['num_questions']} questões | "
          f"OK={s['ok']}  branco={s['blank']}  múltipla={s['multiple']}  revisar={s['review']}")
    print("-" * 58)
    for r in result["results"]:
        ans = r["answer"] or "-"
        fills = " ".join(f"{k}:{v:>5.1f}" for k, v in r["fills"].items())
        print(f"Q{r['question']:>2}  {STATUS_LABEL[r['status']]}  resp={ans:<2}  [{fills}]")


def main():
    ap = argparse.ArgumentParser(description="Leitor OMR de gabaritos (offline).")
    ap.add_argument("images", nargs="+", help="uma ou mais imagens de gabarito")
    ap.add_argument("--debug", action="store_true", help="salva imagem anotada em debug/")
    ap.add_argument("--json", action="store_true", help="imprime JSON bruto")
    args = ap.parse_args()

    os.makedirs("debug", exist_ok=True)
    exit_code = 0

    for path in args.images:
        try:
            if args.debug:
                result, dbg = engine.read_image_file(path, draw_debug=True)
                out = os.path.join("debug", os.path.splitext(os.path.basename(path))[0] + "_debug.png")
                cv2.imwrite(out, dbg)
            else:
                result = engine.read_image_file(path)
        except OMRError as e:
            print(f"\n=== {path} ===\n[ERRO] {e}", file=sys.stderr)
            exit_code = 1
            continue

        if args.json:
            print(json.dumps({"file": path, **result}, ensure_ascii=False, indent=2))
        else:
            print_report(os.path.basename(path), result)
            if args.debug:
                print(f"   (debug salvo em {out})")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
