#!/usr/bin/env python3
"""
Lê fotos de gabarito direto do disco, sem subir a API.

    python test_cli.py foto.jpg                    # descobre o fluxo sozinho
    python test_cli.py foto.jpg --fluxo objetiva   # força o fluxo
    python test_cli.py *.jpg --debug               # salva a foto anotada em debug/
    python test_cli.py foto.jpg --json             # JSON bruto (scripts / conferência)

Fluxos:
    objetiva  -> respostas: nº do aluno + Linguagens e Matemática
    redacao   -> redação:   nº do aluno + quadro de correção do professor

O MODELO da folha (Anos Iniciais x Anos Finais) é reconhecido sozinho e sai no
cabeçalho de cada leitura.
"""
import argparse
import json
import os
import sys

import cv2

from omr import FLUXOS, OMRError, ler_arquivo, ler_fluxo
from omr import template as T

STATUS_LABEL = {
    "OK": "OK       ",
    "BLANK": "EM BRANCO",
    "MULTIPLE": "MULTIPLA ",
    "REVIEW": "REVISAR  ",
}


def _linha_fills(fills: dict) -> str:
    return " ".join(f"{k}:{v:>5.1f}" for k, v in fills.items())


def _imprimir_numero(num: dict) -> None:
    valor = num["value"] or f"(indefinido: {num['raw']})"
    print(f"Número do aluno: {valor}   [{num['status']}]")
    for d in num["digits"]:
        if d["status"] not in ("OK", "BLANK"):
            print(f"   casa {d['position']:>2}  {STATUS_LABEL[d['status']]}  [{_linha_fills(d['fills'])}]")


def imprimir(nome: str, res: dict) -> None:
    a = res["alignment"]
    print(f"\n=== {nome} — fluxo {res['flow']} | modelo {res['model']} ===")
    print(f"registro: {a['fiducials']} | cobertura {a['coverage']:.0%} | "
          f"rotação {a['rotation']}° | ajuste {a['global_fit']}")
    md = a.get("model_detection", {})
    if md:
        print(f"modelo: {md['linhas_detectadas']} linhas de questão, custos {md['custos']}")
    _imprimir_numero(res["student_number"])

    if res["flow"] == "objetiva":
        for sec in res["sections"]:
            s = sec["summary"]
            print(f"\n-- {sec['name'].upper()} ({sec['num_questions']} questões)  "
                  f"OK={s['ok']} branco={s['blank']} múltipla={s['multiple']} revisar={s['review']}")
            for r in sec["results"]:
                print(f"Q{r['question']:>2}  {STATUS_LABEL[r['status']]}  "
                      f"resp={r['answer'] or '-':<2} [{_linha_fills(r['fills'])}]")
    else:
        print("\n-- CORREÇÃO")
        for chave in T.ORDEM_CORRECAO:
            c = res["correction"][chave]
            print(f"{chave:16s} {STATUS_LABEL[c['status']]}  valor={c['value'] or '-':<2} "
                  f"[{_linha_fills(c['fills'])}]")

    s = res["summary"]
    print(f"\nresumo: OK={s['ok']} branco={s['blank']} múltipla={s['multiple']} revisar={s['review']}")


def detectar_fluxo(imagem):
    """Tenta a objetiva; se a folha for a outra página, cai para a redação.

    Devolve (fluxo, resultado_ja_calculado_ou_None) para não ler duas vezes.
    """
    try:
        return "objetiva", ler_fluxo("objetiva", imagem)
    except OMRError:
        return "redacao", None


def main() -> None:
    ap = argparse.ArgumentParser(description="Leitor OMR de gabaritos (offline).")
    ap.add_argument("imagens", nargs="+", help="uma ou mais fotos da folha")
    ap.add_argument("--fluxo", choices=sorted(FLUXOS) + ["auto"], default="auto",
                    help="qual página está sendo lida (padrão: descobre sozinho)")
    ap.add_argument("--debug", action="store_true", help="salva a foto anotada em debug/")
    ap.add_argument("--json", action="store_true", help="imprime o JSON bruto")
    args = ap.parse_args()

    if args.debug:
        os.makedirs("debug", exist_ok=True)
    codigo = 0

    for caminho in args.imagens:
        destino = None
        try:
            imagem = ler_arquivo(caminho)
            if args.fluxo == "auto":
                fluxo, res = detectar_fluxo(imagem)
            else:
                fluxo, res = args.fluxo, None
            if args.debug:
                res, anotada = ler_fluxo(fluxo, imagem, debug=True)
                destino = os.path.join(
                    "debug", os.path.splitext(os.path.basename(caminho))[0] + "_debug.png")
                cv2.imwrite(destino, anotada)
            elif res is None:
                res = ler_fluxo(fluxo, imagem)
        except OMRError as e:
            print(f"\n=== {caminho} ===\n[ERRO] {e}", file=sys.stderr)
            codigo = 1
            continue

        if args.json:
            print(json.dumps({"file": caminho, **res}, ensure_ascii=False, indent=2))
        else:
            imprimir(os.path.basename(caminho), res)
            if destino:
                print(f"   (debug salvo em {destino})")

    sys.exit(codigo)


if __name__ == "__main__":
    main()
