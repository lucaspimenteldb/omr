#!/usr/bin/env python3
"""
Rederiva a geometria de `omr/template.py` a partir do PDF oficial da folha.

É a ferramenta de manutenção do template: se a Editora reimprimir o modelo com
qualquer mudança de layout, rode isto no PDF novo, confira os números e cole-os
em `omr/template.py`. Nada aqui roda em produção.

    # (re)gera os PNGs de referência usados pelos testes — precisa de pymupdf
    python tools/calibrar_template.py --render --pdf "Cartão-Resposta Veloz.pdf"

    # mede os PNGs já existentes e imprime as constantes do template
    python tools/calibrar_template.py

    # confere se o template atual bate com o que está medido no PDF
    python tools/calibrar_template.py --conferir

Como funciona: acha os 4 marcadores fiduciais (as únicas manchas compactas e
quadradas nos cantos), acha todos os círculos impressos, agrupa-os em blocos e
ajusta uma reta a cada eixo de cada bloco. O resíduo desse ajuste é impresso
junto — se ele passar de ~1 px, a folha deixou de ser uma grade regular e o
modelo de `Grade` precisa ser revisto.
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from omr import template as T  # noqa: E402

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELO = os.path.join(RAIZ, "samples", "modelo")
DPI = 300


# --------------------------------------------------------------------------- #
def renderizar(pdf: str) -> None:
    try:
        import fitz
    except ImportError:
        raise SystemExit("--render precisa do pymupdf:  pip install pymupdf")
    os.makedirs(MODELO, exist_ok=True)
    doc = fitz.open(pdf)
    for i, page in enumerate(doc):
        destino = os.path.join(MODELO, f"pagina{i + 1}_{DPI}dpi.png")
        page.get_pixmap(dpi=DPI).save(destino)
        print("gerado:", destino)


# --------------------------------------------------------------------------- #
def achar_fiduciais(img: np.ndarray) -> dict[str, tuple[float, float]]:
    """Centros das 4 marcas de registro, uma por canto."""
    h, w = img.shape
    th = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)[1]
    n, _, stats, cent = cv2.connectedComponentsWithStats(th, 8)
    saida = {}
    for nome, (qx, qy) in {"TL": (0, 0), "TR": (1, 0), "BR": (1, 1), "BL": (0, 1)}.items():
        melhor = None
        for i in range(1, n):
            x, y, bw, bh, area = stats[i]
            cx, cy = cent[i]
            if not (0.015 * w < bw < 0.06 * w) or not (0.010 * h < bh < 0.045 * h):
                continue
            if not (0.85 < bw / bh < 1.18):
                continue
            if not (0.25 < area / (bw * bh) < 0.85):
                continue
            dentro = ((cx < 0.25 * w) if qx == 0 else (cx > 0.75 * w)) and \
                     ((cy < 0.25 * h) if qy == 0 else (cy > 0.75 * h))
            if not dentro:
                continue
            d = (cx - (0 if qx == 0 else w)) ** 2 + (cy - (0 if qy == 0 else h)) ** 2
            if melhor is None or d < melhor[0]:
                melhor = (d, cx, cy)
        if melhor is None:
            raise SystemExit(f"marcador fiducial {nome} não encontrado")
        saida[nome] = (melhor[1], melhor[2])
    return saida


def achar_circulos(img: np.ndarray) -> np.ndarray:
    """(N, 3) x, y, raio de todo círculo impresso. Descarta texto e quadrados."""
    th = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)[1]
    cnts, hier = cv2.findContours(th, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    saida = []
    for c, hz in zip(cnts, hier[0]):
        if hz[3] != -1:
            continue
        a = cv2.contourArea(c)
        if a < 200:
            continue
        (cx, cy), r = cv2.minEnclosingCircle(c)
        p = cv2.arcLength(c, True)
        if 4 * np.pi * a / (p * p + 1e-9) < 0.82 or a / (np.pi * r * r) < 0.75:
            continue
        saida.append((cx, cy, r))
    return np.array(saida).reshape(-1, 3)


def agrupar(vals: np.ndarray, tol: float) -> np.ndarray:
    v = np.sort(np.asarray(vals, float))
    grupos = [[v[0]]]
    for x in v[1:]:
        (grupos[-1].append(x) if x - grupos[-1][-1] <= tol else grupos.append([x]))
    return np.array([np.mean(g) for g in grupos])


def ajustar(centros: np.ndarray) -> tuple[float, float, float]:
    """Ajusta c[i] = a + b*i. Devolve (a, b, resíduo_máximo)."""
    if len(centros) == 1:
        return float(centros[0]), 0.0, 0.0
    idx = np.arange(len(centros))
    b, a = np.polyfit(idx, centros, 1)
    return float(a), float(b), float(np.max(np.abs(centros - (a + b * idx))))


# --------------------------------------------------------------------------- #
class Medidor:
    def __init__(self, pagina: int):
        caminho = os.path.join(MODELO, f"pagina{pagina}_{DPI}dpi.png")
        img = cv2.imread(caminho, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise SystemExit(f"{caminho} não encontrado — rode com --render primeiro")
        self.img = img
        self.fid = achar_fiduciais(img)
        self.x0, self.y0 = self.fid["TL"]
        self.w = self.fid["TR"][0] - self.x0
        self.h = self.fid["BL"][1] - self.y0
        self.circulos = achar_circulos(img)

    def u(self, x):
        return (x - self.x0) / self.w

    def v(self, y):
        return (y - self.y0) / self.h

    def bloco(self, nome: str, caixa, tol=30) -> dict:
        """Mede a grade dentro de (xmin, xmax, ymin, ymax) em px do render."""
        xmin, xmax, ymin, ymax = caixa
        c = self.circulos
        s = c[(c[:, 0] >= xmin) & (c[:, 0] <= xmax) & (c[:, 1] >= ymin) & (c[:, 1] <= ymax)]
        if len(s) == 0:
            raise SystemExit(f"bloco {nome}: nenhum círculo na caixa {caixa}")
        cx, cy = agrupar(s[:, 0], tol), agrupar(s[:, 1], tol)
        ax, bx, rx = ajustar(cx)
        ay, by, ry = ajustar(cy)
        return {
            "nome": nome, "n": len(s), "colunas": len(cx), "linhas": len(cy),
            "u0": self.u(ax), "du": bx / self.w,
            "v0": self.v(ay), "dv": by / self.h,
            "raio": float(np.median(s[:, 2])) / self.w,
            "residuo_px": max(rx, ry),
        }


CAIXAS_P1 = {
    "numero_aluno": (1500, 2450, 480, 1200),
    "linguagens_b1": (200, 650, 2350, 3450),
    "linguagens_b2": (760, 1220, 2350, 3450),
    "matematica_b1": (1390, 1840, 2350, 3450),
    "matematica_b2": (1960, 2410, 2350, 3450),
}
CAIXAS_P2 = {
    "numero_aluno": (1500, 2450, 480, 1200),
    "correcao_g1": (500, 900, 3300, 3500),
    "correcao_g2": (1240, 1620, 3300, 3500),
    "correcao_g3": (1950, 2340, 3300, 3500),
}

# (bloco medido) -> (bloco do template). O template não tem um objeto por
# medição, então comparamos com a Grade correspondente.
GRADES_TEMPLATE = {
    "numero_aluno": T.GRADE_NUMERO_ALUNO,
    "linguagens_b1": T.LINGUAGENS_B1.grade,
    "linguagens_b2": T.LINGUAGENS_B2.grade,
    "matematica_b1": T.MATEMATICA_B1.grade,
    "matematica_b2": T.MATEMATICA_B2.grade,
    "correcao_g1": T.CORRECAO_G1.grade,
    "correcao_g2": T.CORRECAO_G2.grade,
    "correcao_g3": T.CORRECAO_G3.grade,
}


def medir_tudo() -> list[dict]:
    saida = []
    for pagina, caixas in ((1, CAIXAS_P1), (2, CAIXAS_P2)):
        med = Medidor(pagina)
        print(f"\n### página {pagina} — quadro fiducial {med.w:.1f} x {med.h:.1f} px "
              f"(aspecto {med.h / med.w:.6f}; template usa {T.ASPECTO:.6f})")
        for nome, caixa in caixas.items():
            b = med.bloco(nome, caixa)
            b["pagina"] = pagina
            saida.append(b)
            print(f"  {nome:15s} {b['linhas']:>2}x{b['colunas']:<2} bolhas={b['n']:>3}  "
                  f"u0={b['u0']:.6f} du={b['du']:.6f}  v0={b['v0']:.6f} dv={b['dv']:.6f}  "
                  f"raio={b['raio']:.6f}  resíduo={b['residuo_px']:.2f}px")
    return saida


def conferir(medidos: list[dict], tol_px: float = 2.0) -> int:
    """Compara o medido com `omr/template.py`. Devolve o nº de divergências."""
    print(f"\n### conferência (tolerância {tol_px} px no canônico de "
          f"{T.FIDUCIAL_W_PX:.0f} px de largura)")
    ruins = 0
    for b in medidos:
        if b["pagina"] == 2 and b["nome"] == "numero_aluno":
            continue                       # mesma grade da página 1
        g = GRADES_TEMPLATE[b["nome"]]
        difs = {
            "u0": (b["u0"] - g.u0) * T.FIDUCIAL_W_PX,
            "du": (b["du"] - g.du) * T.FIDUCIAL_W_PX,
            "v0": (b["v0"] - g.v0) * T.FIDUCIAL_H_PX,
            "dv": (b["dv"] - g.dv) * T.FIDUCIAL_H_PX,
            "raio": (b["raio"] - g.raio) * T.FIDUCIAL_W_PX,
        }
        pior = max(abs(v) for v in difs.values())
        forma_ok = (g.n_linhas, g.n_colunas) == (b["linhas"], b["colunas"])
        marca = "ok " if pior <= tol_px and forma_ok else "!! "
        if marca == "!! ":
            ruins += 1
        detalhe = " ".join(f"{k}={v:+.2f}" for k, v in difs.items())
        forma = "" if forma_ok else f"  FORMA template={g.n_linhas}x{g.n_colunas}"
        print(f"  {marca}{b['nome']:15s} pior={pior:5.2f}px  {detalhe}{forma}")
    print("  => template em dia" if not ruins else f"  => {ruins} bloco(s) divergentes")
    return ruins


def main() -> None:
    ap = argparse.ArgumentParser(description="Calibra/confere o template da folha.")
    ap.add_argument("--render", action="store_true", help="rasteriza o PDF em samples/modelo/")
    ap.add_argument("--pdf", help="caminho do PDF oficial (com --render)")
    ap.add_argument("--conferir", action="store_true",
                    help="compara o medido com omr/template.py e sai != 0 se divergir")
    args = ap.parse_args()

    if args.render:
        if not args.pdf:
            raise SystemExit("--render precisa de --pdf CAMINHO")
        renderizar(args.pdf)

    medidos = medir_tudo()
    if args.conferir:
        sys.exit(1 if conferir(medidos) else 0)
    conferir(medidos)


if __name__ == "__main__":
    main()
