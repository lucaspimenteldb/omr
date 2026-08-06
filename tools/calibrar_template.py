#!/usr/bin/env python3
"""
Rederiva a geometria de `omr/template.py` a partir do PDF oficial da folha.

É a ferramenta de manutenção do template: se a Editora reimprimir o modelo com
qualquer mudança de layout, rode isto no PDF novo, confira os números e cole-os
em `omr/template.py`. Nada aqui roda em produção.

    # rasteriza o PDF oficial de um modelo (só a página que interessa)
    python tools/calibrar_template.py --render --pdf "Anos Iniciais.pdf" \\
           --modelo anos_iniciais --pagina 1 --rotacao 180

    # mede os PNGs já existentes e imprime as constantes do template
    python tools/calibrar_template.py

    # confere se o template atual bate com o PDF (sai != 0 se divergir)
    python tools/calibrar_template.py --conferir

O PDF da coleção traz várias páginas e algumas vêm giradas 180°; por isso
`--pagina` e `--rotacao` existem. A rotação é aplicada ANTES de qualquer
medição — medir de cabeça para baixo espelharia todas as coordenadas.

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
#: nome do PNG de referência de cada (modelo, fluxo) em samples/modelo/.
RENDERS = {
    ("anos_finais", "objetiva"): "pagina1_300dpi.png",
    ("anos_finais", "redacao"): "pagina2_300dpi.png",
    ("anos_iniciais", "objetiva"): "anos_iniciais_objetiva_300dpi.png",
}


def renderizar(pdf: str, modelo: str, fluxo: str, pagina: int, rotacao: int) -> str:
    """Rasteriza UMA página do PDF, já endireitada, no destino do modelo."""
    try:
        import fitz
    except ImportError:
        raise SystemExit("--render precisa do pymupdf:  pip install pymupdf")
    nome = RENDERS.get((modelo, fluxo))
    if nome is None:
        raise SystemExit(f"não sei onde guardar o render de {modelo}/{fluxo}")

    os.makedirs(MODELO, exist_ok=True)
    doc = fitz.open(pdf)
    if not 1 <= pagina <= doc.page_count:
        raise SystemExit(f"--pagina {pagina} fora de 1..{doc.page_count}")
    destino = os.path.join(MODELO, nome)

    pix = doc[pagina - 1].get_pixmap(dpi=DPI)
    img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR if pix.n == 3 else cv2.COLOR_RGBA2BGR)
    giro = {0: None, 90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180,
            270: cv2.ROTATE_90_COUNTERCLOCKWISE}
    if rotacao not in giro:
        raise SystemExit("--rotacao deve ser 0, 90, 180 ou 270")
    if giro[rotacao] is not None:
        img = cv2.rotate(img, giro[rotacao])
    cv2.imwrite(destino, img)
    print(f"gerado: {destino}  ({img.shape[1]}x{img.shape[0]} px, página {pagina} "
          f"de {doc.page_count}, girada {rotacao}°)")
    return destino


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
    """Mede um render de referência no quadro fiducial daquele render."""

    def __init__(self, modelo: str, fluxo: str):
        nome = RENDERS.get((modelo, fluxo))
        if nome is None:
            raise SystemExit(f"não há render previsto para {modelo}/{fluxo}")
        caminho = os.path.join(MODELO, nome)
        img = cv2.imread(caminho, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise SystemExit(
                f"{caminho} não encontrado — rasterize primeiro:\n"
                f"  python tools/calibrar_template.py --render --pdf ARQ.pdf "
                f"--modelo {modelo} --fluxo {fluxo} [--pagina N] [--rotacao 180]"
            )
        self.modelo, self.fluxo, self.caminho = modelo, fluxo, caminho
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

    def _caixa_do_bloco(self, bloco: T.Bloco, folga: float = 0.45):
        """Onde procurar as bolhas desse bloco, em px do render.

        A caixa sai do PRÓPRIO template — assim o calibrador acompanha qualquer
        modelo novo sem lista de coordenadas escrita à mão. A folga é MEIO PASSO
        (menos um tico): mais que isso e a caixa engole a primeira coluna do
        bloco vizinho ou a linha de cima, e a medição sai com forma errada.
        """
        g = bloco.grade
        raio_v = g.raio * T.FIDUCIAL_W_PX / T.FIDUCIAL_H_PX
        pad_u = folga * g.du if g.n_colunas > 1 else 1.5 * g.raio
        pad_v = folga * g.dv if g.n_linhas > 1 else 1.5 * raio_v
        pad_u, pad_v = max(pad_u, g.raio * 1.02), max(pad_v, raio_v * 1.02)
        u_min, u_max = g.u0 - pad_u, g.u0 + (g.n_colunas - 1) * g.du + pad_u
        v_min, v_max = g.v0 - pad_v, g.v0 + (g.n_linhas - 1) * g.dv + pad_v
        return (self.x0 + u_min * self.w, self.x0 + u_max * self.w,
                self.y0 + v_min * self.h, self.y0 + v_max * self.h)

    def bloco(self, bloco: T.Bloco, tol=30) -> dict:
        xmin, xmax, ymin, ymax = self._caixa_do_bloco(bloco)
        c = self.circulos
        s = c[(c[:, 0] >= xmin) & (c[:, 0] <= xmax) & (c[:, 1] >= ymin) & (c[:, 1] <= ymax)]
        if len(s) == 0:
            raise SystemExit(f"bloco {bloco.nome}: nenhum círculo onde o template espera")
        cx, cy = agrupar(s[:, 0], tol), agrupar(s[:, 1], tol)
        ax, bx, rx = ajustar(cx)
        ay, by, ry = ajustar(cy)
        return {
            "modelo": self.modelo, "fluxo": self.fluxo, "nome": bloco.nome,
            "n": len(s), "colunas": len(cx), "linhas": len(cy),
            "u0": self.u(ax), "du": bx / self.w,
            "v0": self.v(ay), "dv": by / self.h,
            "raio": float(np.median(s[:, 2])) / self.w,
            "residuo_px": max(rx, ry),
            "grade": bloco.grade,
        }


def medir_tudo(apenas: str | None = None) -> list[dict]:
    """Mede todos os renders disponíveis. `apenas` filtra por modelo."""
    saida = []
    for (modelo, fluxo), nome in RENDERS.items():
        if apenas and modelo != apenas:
            continue
        if not os.path.exists(os.path.join(MODELO, nome)):
            print(f"\n### {modelo}/{fluxo}: render ausente ({nome}) — pulando")
            continue
        med = Medidor(modelo, fluxo)
        folha = T.MODELOS[modelo].folhas[fluxo]
        print(f"\n### {modelo}/{fluxo} — quadro fiducial {med.w:.1f} x {med.h:.1f} px "
              f"(aspecto {med.h / med.w:.6f}; template usa {T.ASPECTO:.6f})")
        for bloco in folha.blocos:
            b = med.bloco(bloco)
            saida.append(b)
            print(f"  {bloco.nome:15s} {b['linhas']:>2}x{b['colunas']:<2} bolhas={b['n']:>3}  "
                  f"u0={b['u0']:.6f} du={b['du']:.6f}  v0={b['v0']:.6f} dv={b['dv']:.6f}  "
                  f"raio={b['raio']:.6f}  resíduo={b['residuo_px']:.2f}px")
    return saida


def conferir(medidos: list[dict], tol_px: float = 2.0) -> int:
    """Compara o medido com `omr/template.py`. Devolve o nº de divergências."""
    print(f"\n### conferência (tolerância {tol_px} px no canônico de "
          f"{T.FIDUCIAL_W_PX:.0f} px de largura)")
    ruins = 0
    vistos = set()
    for b in medidos:
        chave = (b["modelo"], b["nome"])
        if chave in vistos:
            continue                       # o nº do aluno se repete entre páginas
        vistos.add(chave)
        g = b["grade"]
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
        print(f"  {marca}{b['modelo']:14s} {b['nome']:15s} pior={pior:5.2f}px  {detalhe}{forma}")
    print("  => template em dia" if not ruins else f"  => {ruins} bloco(s) divergentes")
    return ruins


def main() -> None:
    ap = argparse.ArgumentParser(description="Calibra/confere o template da folha.")
    ap.add_argument("--render", action="store_true", help="rasteriza o PDF em samples/modelo/")
    ap.add_argument("--pdf", help="caminho do PDF oficial (com --render)")
    ap.add_argument("--modelo", choices=sorted(T.MODELOS), default="anos_finais",
                    help="qual modelo esse PDF é")
    ap.add_argument("--fluxo", choices=("objetiva", "redacao"), default="objetiva",
                    help="qual página do modelo")
    ap.add_argument("--pagina", type=int, default=1,
                    help="número da página dentro do PDF (1-based)")
    ap.add_argument("--rotacao", type=int, default=0, choices=(0, 90, 180, 270),
                    help="giro aplicado ANTES de medir (páginas vêm de cabeça para baixo)")
    ap.add_argument("--conferir", action="store_true",
                    help="compara com omr/template.py e sai != 0 se divergir")
    args = ap.parse_args()

    if args.render:
        if not args.pdf:
            raise SystemExit("--render precisa de --pdf CAMINHO")
        renderizar(args.pdf, args.modelo, args.fluxo, args.pagina, args.rotacao)

    medidos = medir_tudo(args.modelo if args.render else None)
    ruins = conferir(medidos)
    if args.conferir:
        sys.exit(1 if ruins else 0)


if __name__ == "__main__":
    main()
