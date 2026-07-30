"""
Núcleo de leitura: mede e classifica as bolhas de uma folha já registrada.

Recebe a imagem canônica (saída de `omr/registration.py`) e um `Bloco` do
template, e faz, para cada bloco:

  1. **reancora a grade nas bolhas reais.** As posições do template são só o
     palpite inicial: dentro do recorte do bloco procuramos as bolhas
     efetivamente impressas e casamos cada uma com a posição esperada. Um ajuste
     afim global (com todos os blocos juntos) absorve escala/rotação residual e
     um deslocamento local por bloco absorve a curvatura do papel;
  2. **mede** o percentual de pixels escuros dentro de cada bolha, num disco
     menor que o círculo impresso (para não contar o anel nem o dígito impresso);
  3. **classifica** cada campo em OK / BLANK / MULTIPLE / REVIEW.

Se o reajuste não for confiável (poucas bolhas detectadas, escala absurda), o
bloco cai de volta para as posições do template em vez de chutar — errar por
não-ajustar é recuperável, errar por ajustar torto não é.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from . import config as C
from . import template as T
from .registration import CANON_IMG_H, CANON_IMG_W, OMRError, para_canonico, raio_canonico

__all__ = [
    "OMRError", "ResultadoCampo", "ResultadoBloco", "LeituraFolha",
    "ler_folha", "sondar_objetiva", "centros_do_bloco",
]


# --------------------------------------------------------------------------- #
# Estruturas de resultado
# --------------------------------------------------------------------------- #
@dataclass
class ResultadoCampo:
    """Uma questão, um dígito do número do aluno ou uma competência."""

    chave: str
    valor: str | None
    status: str                       # OK | BLANK | MULTIPLE | REVIEW
    marcadas: list[str]
    fills: dict[str, float]


@dataclass
class ResultadoBloco:
    nome: str
    campos: list[ResultadoCampo]
    centros: np.ndarray               # (n_linhas, n_colunas, 2) em px canônicos
    raio_px: float
    alinhamento: str                  # "afim+local" | "afim" | "local" | "template"
    pares: int                        # bolhas casadas com a grade
    esperadas: int                    # bolhas que o template prevê


@dataclass
class LeituraFolha:
    blocos: dict[str, ResultadoBloco]
    ajuste_global: str                # "afim" | "identidade"
    pares_globais: int
    esperadas_globais: int
    detalhes: dict = field(default_factory=dict)

    @property
    def cobertura(self) -> float:
        """Fração das bolhas do template que foram encontradas na foto."""
        if not self.esperadas_globais:
            return 0.0
        return self.pares_globais / self.esperadas_globais


# --------------------------------------------------------------------------- #
# Geometria dos blocos no espaço canônico
# --------------------------------------------------------------------------- #
def centros_do_bloco(bloco: T.Bloco) -> np.ndarray:
    """Centros esperados (n_linhas, n_colunas, 2) em pixels canônicos."""
    g = bloco.grade
    pts = np.empty((g.n_linhas, g.n_colunas, 2), dtype="float64")
    for li in range(g.n_linhas):
        for ci in range(g.n_colunas):
            pts[li, ci] = para_canonico(*g.centro(li, ci))
    return pts


def _passos_px(bloco: T.Bloco) -> tuple[float, float]:
    """Passo horizontal e vertical do bloco, em px canônicos."""
    g = bloco.grade
    a = np.array(para_canonico(g.u0, g.v0))
    du = abs(np.array(para_canonico(g.u0 + g.du, g.v0))[0] - a[0]) if g.n_colunas > 1 else 0.0
    dv = abs(np.array(para_canonico(g.u0, g.v0 + g.dv))[1] - a[1]) if g.n_linhas > 1 else 0.0
    return du, dv


def _passo_minimo(bloco: T.Bloco) -> float:
    du, dv = _passos_px(bloco)
    vals = [p for p in (du, dv) if p > 0]
    return min(vals) if vals else 4 * raio_canonico(bloco.grade.raio)


# --------------------------------------------------------------------------- #
# Pré-processamento
# --------------------------------------------------------------------------- #
def _flatfield(img: np.ndarray) -> np.ndarray:
    """Divide pelo próprio borrão: mata sombra, gradiente de luz e tarja
    colorida. Depois disso o papel fica ~255 e a tinta bem abaixo."""
    bg = cv2.GaussianBlur(img, (0, 0), C.FLATFIELD_SIGMA)
    return cv2.divide(img, bg, scale=255).astype("uint8")


def _recorte(canon: np.ndarray, centros: np.ndarray, raio_px: float) -> tuple[np.ndarray, int, int]:
    """Recorta o ROI do bloco com folga. Devolve (roi_normalizado, x0, y0)."""
    pad = 2.5 * raio_px
    xs, ys = centros[..., 0], centros[..., 1]
    x0 = int(max(0, np.floor(xs.min() - pad)))
    y0 = int(max(0, np.floor(ys.min() - pad)))
    x1 = int(min(CANON_IMG_W, np.ceil(xs.max() + pad)))
    y1 = int(min(CANON_IMG_H, np.ceil(ys.max() + pad)))
    if x1 - x0 < 8 or y1 - y0 < 8:
        raise OMRError("Bloco fora da área útil da folha após o registro.")
    return _flatfield(canon[y0:y1, x0:x1]), x0, y0


def _detectar_bolhas(roi: np.ndarray, raio_px: float) -> np.ndarray:
    """Acha candidatas a bolha no recorte já normalizado. Devolve (N, 3): x,y,r.

    Combina duas máscaras porque os dois tipos de bolha aparecem diferente:
    a vazia é um anel fino (o adaptativo pega bem) e a marcada é um disco
    maciço (o limiar fixo pega bem; o adaptativo esvazia o miolo dela).
    """
    adapt = cv2.adaptiveThreshold(
        roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
        C.ADAPT_BLOCK, C.ADAPT_C,
    )
    fixo = cv2.threshold(roi, C.NIVEL_TINTA, 255, cv2.THRESH_BINARY_INV)[1]
    mask = cv2.bitwise_or(adapt, fixo)

    r_min = C.RAIO_MIN_FRAC * raio_px
    r_max = C.RAIO_MAX_FRAC * raio_px
    area_min = 0.35 * np.pi * r_min * r_min

    cnts, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    saida = []
    for c, hz in zip(cnts, hier[0] if hier is not None else []):
        if hz[3] != -1:                       # só contornos externos
            continue
        area = cv2.contourArea(c)
        if area < area_min:
            continue
        (cx, cy), r = cv2.minEnclosingCircle(c)
        if not (r_min <= r <= r_max):
            continue
        peri = cv2.arcLength(c, True)
        if peri <= 0:
            continue
        if 4 * np.pi * area / (peri * peri) < C.CIRCULARIDADE_MIN:
            continue
        if area / (np.pi * r * r) < C.PREENCH_CIRCULO_MIN:
            continue
        saida.append((cx, cy, r))
    return np.array(saida, dtype="float64").reshape(-1, 3)


# --------------------------------------------------------------------------- #
# Reajuste da grade
# --------------------------------------------------------------------------- #
def _casar(esperados: np.ndarray, candidatas: np.ndarray, tol: float) -> tuple[np.ndarray, np.ndarray]:
    """Casa cada posição esperada com no máximo uma candidata, do par mais
    próximo para o mais distante. Devolve (esperados_casados, observados)."""
    if len(esperados) == 0 or len(candidatas) == 0:
        return np.empty((0, 2)), np.empty((0, 2))
    d = np.linalg.norm(esperados[:, None, :] - candidatas[None, :, :2], axis=2)
    ordem = np.dstack(np.unravel_index(np.argsort(d, axis=None), d.shape))[0]
    usados_e: set[int] = set()
    usados_c: set[int] = set()
    pares = []
    for i, j in ordem:
        if d[i, j] > tol:
            break
        if i in usados_e or j in usados_c:
            continue
        usados_e.add(int(i))
        usados_c.add(int(j))
        pares.append((int(i), int(j)))
    if not pares:
        return np.empty((0, 2)), np.empty((0, 2))
    ei = np.array([p[0] for p in pares])
    cj = np.array([p[1] for p in pares])
    return esperados[ei], candidatas[cj, :2]


def _fit_afim(src: np.ndarray, dst: np.ndarray) -> np.ndarray | None:
    """Mínimos quadrados de src -> dst com duas rodadas de poda de outlier."""
    if len(src) < 6:
        return None
    s, d = src, dst
    A = None
    for _ in range(3):
        M = np.hstack([s, np.ones((len(s), 1))])
        sol, *_ = np.linalg.lstsq(M, d, rcond=None)
        A = sol.T                                     # (2, 3)
        res = np.linalg.norm(M @ sol - d, axis=1)
        corte = max(2.5 * float(np.median(res)), 1.0)
        manter = res <= corte
        if manter.all() or manter.sum() < 6:
            break
        s, d = s[manter], d[manter]
    return A


def _afim_sensata(A: np.ndarray) -> bool:
    """Rejeita ajustes que esticam, espelham ou giram demais a folha."""
    if A is None or not np.all(np.isfinite(A)):
        return False
    L = A[:, :2]
    det = float(np.linalg.det(L))
    if det <= 0:                                      # espelhamento
        return False
    sx = float(np.hypot(L[0, 0], L[1, 0]))
    sy = float(np.hypot(L[0, 1], L[1, 1]))
    if abs(sx - 1) > C.SNAP_ESCALA_TOL or abs(sy - 1) > C.SNAP_ESCALA_TOL:
        return False
    # cisalhamento / rotação: colunas devem seguir quase ortogonais aos eixos
    if abs(L[1, 0]) > 0.12 * sx or abs(L[0, 1]) > 0.12 * sy:
        return False
    return True


def _aplicar_afim(A: np.ndarray, pts: np.ndarray) -> np.ndarray:
    forma = pts.shape
    p = pts.reshape(-1, 2)
    out = p @ A[:, :2].T + A[:, 2]
    return out.reshape(forma)


# --------------------------------------------------------------------------- #
# Medição e classificação
# --------------------------------------------------------------------------- #
def _medir_fills(roi: np.ndarray, centros_locais: np.ndarray, raio_px: float) -> np.ndarray:
    """% de pixels escuros dentro de cada bolha (disco de amostragem)."""
    rs = max(2, int(round(C.AMOSTRA_R_FRAC * raio_px)))
    mask = np.zeros((2 * rs + 1, 2 * rs + 1), np.uint8)
    cv2.circle(mask, (rs, rs), rs, 255, -1)
    area = int(cv2.countNonZero(mask))
    escuro = (roi < C.NIVEL_TINTA).astype(np.uint8) * 255

    h, w = roi.shape
    saida = np.zeros(centros_locais.shape[:-1], dtype="float64")
    it = np.ndindex(*centros_locais.shape[:-1])
    for idx in it:
        cx, cy = centros_locais[idx]
        x0, y0 = int(round(cx)) - rs, int(round(cy)) - rs
        if x0 < 0 or y0 < 0 or x0 + 2 * rs + 1 > w or y0 + 2 * rs + 1 > h:
            saida[idx] = 0.0
            continue
        patch = escuro[y0 : y0 + 2 * rs + 1, x0 : x0 + 2 * rs + 1]
        saida[idx] = 100.0 * cv2.countNonZero(cv2.bitwise_and(patch, mask)) / area
    return saida


def _classificar(fills: dict[str, float]) -> tuple[str | None, str]:
    """(valor, status) de um campo a partir do preenchimento de cada opção."""
    marcadas = [k for k, f in fills.items() if f >= C.MARK_THRESHOLD]
    ambiguas = [k for k, f in fills.items() if C.REVIEW_LOW <= f < C.MARK_THRESHOLD]
    if len(marcadas) >= 2:
        return None, "MULTIPLE"
    if ambiguas:
        return (marcadas[0] if marcadas else None), "REVIEW"
    if len(marcadas) == 1:
        return marcadas[0], "OK"
    return None, "BLANK"


# --------------------------------------------------------------------------- #
# Leitura de um bloco
# --------------------------------------------------------------------------- #
@dataclass
class _Preparo:
    bloco: T.Bloco
    esperados: np.ndarray             # (n_lin, n_col, 2) no canônico
    roi: np.ndarray
    x0: int
    y0: int
    candidatas: np.ndarray            # (N, 3) em coordenadas do ROI
    raio_px: float


def _preparar(canon: np.ndarray, bloco: T.Bloco) -> _Preparo:
    esperados = centros_do_bloco(bloco)
    raio_px = raio_canonico(bloco.grade.raio)
    roi, x0, y0 = _recorte(canon, esperados, raio_px)
    cand = _detectar_bolhas(roi, raio_px)
    return _Preparo(bloco, esperados, roi, x0, y0, cand, raio_px)


def _pares_do_preparo(p: _Preparo, esperados: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Casa as candidatas do bloco com as posições esperadas (no canônico)."""
    if len(p.candidatas) == 0:
        return np.empty((0, 2)), np.empty((0, 2))
    cand_canon = p.candidatas.copy()
    cand_canon[:, 0] += p.x0
    cand_canon[:, 1] += p.y0
    tol = C.SNAP_TOL_FRAC * _passo_minimo(p.bloco)
    return _casar(esperados.reshape(-1, 2), cand_canon, tol)


def _ler_bloco(p: _Preparo, centros: np.ndarray, alinhamento: str, pares: int) -> ResultadoBloco:
    bloco = p.bloco
    locais = centros.copy()
    locais[..., 0] -= p.x0
    locais[..., 1] -= p.y0
    fills = _medir_fills(p.roi, locais, p.raio_px)

    campos: list[ResultadoCampo] = []
    for i, chave in enumerate(bloco.chaves):
        rotulos = bloco.rotulos_de(i)
        celulas = bloco.celulas_do_campo(i)
        f = {rot: round(float(fills[li, ci]), 1) for rot, (li, ci) in zip(rotulos, celulas)}
        valor, status = _classificar(f)
        campos.append(
            ResultadoCampo(
                chave=chave,
                valor=valor,
                status=status,
                marcadas=[k for k, v in f.items() if v >= C.MARK_THRESHOLD],
                fills=f,
            )
        )
    return ResultadoBloco(
        nome=bloco.nome,
        campos=campos,
        centros=centros,
        raio_px=p.raio_px,
        alinhamento=alinhamento,
        pares=pares,
        esperadas=int(np.prod(p.esperados.shape[:-1])),
    )


# --------------------------------------------------------------------------- #
# Leitura da folha inteira
# --------------------------------------------------------------------------- #
def ler_folha(canon: np.ndarray, folha: T.Folha) -> LeituraFolha:
    """Lê todos os blocos de uma folha na imagem canônica."""
    preparos = [_preparar(canon, b) for b in folha.blocos]

    # --- ajuste afim global: junta os pares de TODOS os blocos ---------------
    src_all, dst_all = [], []
    for p in preparos:
        e, o = _pares_do_preparo(p, p.esperados)
        if len(e):
            src_all.append(e)
            dst_all.append(o)
    A = None
    if src_all:
        src = np.vstack(src_all)
        dst = np.vstack(dst_all)
        if len(src) >= C.SNAP_MIN_PARES_PAGINA:
            cand = _fit_afim(src, dst)
            A = cand if _afim_sensata(cand) else None

    blocos: dict[str, ResultadoBloco] = {}
    pares_total = 0
    esperadas_total = 0
    for p in preparos:
        esperados = _aplicar_afim(A, p.esperados) if A is not None else p.esperados
        e, o = _pares_do_preparo(p, esperados)

        alinhamento = "afim" if A is not None else "template"
        centros = esperados
        if len(e) >= C.SNAP_MIN_PARES_BLOCO:
            desloc = np.median(o - e, axis=0)
            limite = C.SNAP_DESLOC_MAX_FRAC * _passo_minimo(p.bloco)
            if float(np.hypot(*desloc)) <= limite:
                centros = esperados + desloc
                alinhamento = "afim+local" if A is not None else "local"

        res = _ler_bloco(p, centros, alinhamento, len(e))
        blocos[p.bloco.nome] = res
        pares_total += len(e)
        esperadas_total += res.esperadas

    return LeituraFolha(
        blocos=blocos,
        ajuste_global="afim" if A is not None else "identidade",
        pares_globais=pares_total,
        esperadas_globais=esperadas_total,
    )


def sondar_objetiva(canon: np.ndarray) -> int:
    """Quantas bolhas de questão aparecem onde a página objetiva as prevê.

    Serve para dois usos: distinguir a página 1 da página 2 e detectar uma foto
    de cabeça para baixo (aí a contagem despenca).
    """
    total = 0
    for bloco in (T.LINGUAGENS_B1, T.LINGUAGENS_B2, T.MATEMATICA_B1, T.MATEMATICA_B2):
        try:
            p = _preparar(canon, bloco)
        except OMRError:
            continue
        e, _ = _pares_do_preparo(p, p.esperados)
        total += len(e)
    return total
