"""
Registro da folha: da foto crua para a "folha canônica".

A folha tem 4 marcadores fiduciais impressos (cruz + quadradinho, um em cada
canto). Eles são a referência de perspectiva: achando os 4 centros, uma
homografia leva a foto — torta, inclinada, fotografada de qualquer ângulo — para
um espaço canônico onde as coordenadas normalizadas de `omr/template.py` valem
exatamente.

O registro é feito em camadas, cada uma tolerante à falha da anterior:

  1. acha o PAPEL na foto (maior quadrilátero) e retifica grosseiramente.
     Falhou? Assume que a foto já é um recorte da folha.
  2. procura cada MARCA nos 4 cantos por template matching multiescala.
     Achou 3? Infere a 4ª por paralelogramo.
     Achou menos? Cai para os cantos do papel (registro aproximado).
  3. monta a homografia foto -> canônica e devolve a folha retificada.

O erro residual que sobrar depois disso é corrigido bloco a bloco em
`omr/engine.py`, que reancora a grade nas bolhas realmente impressas.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from . import config as C
from . import template as T


class OMRError(Exception):
    """Falha estrutural: não foi possível ler a folha."""


class EntradaInvalida(OMRError):
    """A imagem em si não serve — girar ou reenquadrar não resolveria.

    Separada de `OMRError` para que a busca de orientação em `omr/flows.py` não
    engula a mensagem específica tentando as outras três rotações.
    """


# --------------------------------------------------------------------------- #
# Espaço canônico
# --------------------------------------------------------------------------- #
def _dim_canonica() -> tuple[int, int, int, int]:
    """(largura_img, altura_img, margem_px, altura_do_quadro_px)."""
    margem = int(round(C.CANON_W * C.CANON_MARGEM))
    quadro_h = int(round(C.CANON_W * T.ASPECTO))
    return C.CANON_W + 2 * margem, quadro_h + 2 * margem, margem, quadro_h


CANON_IMG_W, CANON_IMG_H, CANON_MARGEM_PX, CANON_QUADRO_H = _dim_canonica()


def para_canonico(u: float, v: float) -> tuple[float, float]:
    """(u, v) normalizado no quadro fiducial -> pixel da imagem canônica."""
    return (CANON_MARGEM_PX + u * C.CANON_W, CANON_MARGEM_PX + v * CANON_QUADRO_H)


def raio_canonico(raio_norm: float) -> float:
    """Raio normalizado (fração da largura do quadro) -> raio em px."""
    return raio_norm * C.CANON_W


#: Cantos do quadro fiducial na imagem canônica, na ordem TL, TR, BR, BL.
CANTOS_CANONICOS = np.array(
    [para_canonico(0, 0), para_canonico(1, 0), para_canonico(1, 1), para_canonico(0, 1)],
    dtype="float32",
)


@dataclass
class Registro:
    """Resultado do registro."""

    canonica: np.ndarray            # imagem cinza retificada (CANON_IMG_H, CANON_IMG_W)
    homografia: np.ndarray          # 3x3: foto original -> canônica
    origem_fiduciais: str           # "fiduciais" | "fiduciais(3+1)" | "papel"
    scores: list[float]             # correlação de cada marca (vazio no fallback)
    papel_detectado: bool

    def para_original(self, pts: np.ndarray) -> np.ndarray:
        """Leva pontos do espaço canônico de volta para a foto original."""
        inv = np.linalg.inv(self.homografia)
        p = np.asarray(pts, dtype="float32").reshape(-1, 1, 2)
        return cv2.perspectiveTransform(p, inv).reshape(-1, 2)


# --------------------------------------------------------------------------- #
# Utilitários geométricos
# --------------------------------------------------------------------------- #
def _ordenar_cantos(pts: np.ndarray) -> np.ndarray:
    """Ordena 4 pontos como [sup-esq, sup-dir, inf-dir, inf-esq]."""
    pts = np.asarray(pts, dtype="float32").reshape(4, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array(
        [pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]],
        dtype="float32",
    )


def _lado(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.hypot(*(np.asarray(a) - np.asarray(b))))


# --------------------------------------------------------------------------- #
# Orientação da foto
# --------------------------------------------------------------------------- #
#: Rotações suportadas, em graus no sentido horário.
ROTACOES = (0, 90, 180, 270)

_CV_ROTACAO = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


def _matriz_rotacao(rotacao: int, w: int, h: int) -> np.ndarray:
    """Leva um ponto da foto ORIGINAL (w x h) para a foto já rotacionada.

    As fórmulas seguem a convenção do `cv2.rotate`: girando 90° no sentido
    horário, a coluna x vira a linha x e a linha y vira a coluna (h-1-y).
    """
    if rotacao == 90:
        return np.array([[0, -1, h - 1], [1, 0, 0], [0, 0, 1]], "float64")
    if rotacao == 180:
        return np.array([[-1, 0, w - 1], [0, -1, h - 1], [0, 0, 1]], "float64")
    if rotacao == 270:
        return np.array([[0, 1, 0], [-1, 0, w - 1], [0, 0, 1]], "float64")
    return np.eye(3)


# --------------------------------------------------------------------------- #
# Etapa 1 — localizar o papel
# --------------------------------------------------------------------------- #
def _achar_papel(gray: np.ndarray) -> np.ndarray | None:
    """Maior quadrilátero convexo plausível = a folha. None se não achar."""
    h, w = gray.shape
    esc = C.PAPEL_LARGURA_TRABALHO / float(max(h, w))
    esc = min(esc, 1.0)
    peq = cv2.resize(gray, None, fx=esc, fy=esc, interpolation=cv2.INTER_AREA)
    ph, pw = peq.shape
    area_img = float(ph * pw)
    borrada = cv2.GaussianBlur(peq, (5, 5), 0)

    candidatos: list[np.ndarray] = []
    # (a) papel claro contra fundo mais escuro
    mascaras = [cv2.threshold(borrada, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]]
    # (b) bordas — funciona quando o fundo também é claro
    bordas = cv2.Canny(borrada, 40, 130)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    mascaras.append(cv2.morphologyEx(bordas, cv2.MORPH_CLOSE, k, iterations=2))

    for m in mascaras:
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:5]:
            area = cv2.contourArea(c)
            if area < C.PAPEL_AREA_MIN_FRAC * area_img:
                continue
            peri = cv2.arcLength(c, True)
            for eps in (0.02, 0.035, 0.05):
                approx = cv2.approxPolyDP(c, eps * peri, True)
                if len(approx) == 4 and cv2.isContourConvex(approx):
                    candidatos.append((_ordenar_cantos(approx) / esc).astype("float32"))
                    break

    if not candidatos:
        return None
    # o maior deles, desde que não seja praticamente a foto inteira
    candidatos.sort(key=lambda q: cv2.contourArea(q.reshape(-1, 1, 2)), reverse=True)
    melhor = candidatos[0]
    if cv2.contourArea(melhor.reshape(-1, 1, 2)) > 0.985 * (h * w):
        return None
    return melhor


def _retificar_pagina(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
    """Retifica a página grosseiramente. Devolve (página, M_foto->página, achou)."""
    larg = C.PAGINA_LARGURA
    alt = int(round(larg * T.PAGINA_ASPECTO))
    dst = np.array([[0, 0], [larg, 0], [larg, alt], [0, alt]], "float32")

    quad = _achar_papel(gray)
    if quad is not None:
        M = cv2.getPerspectiveTransform(quad, dst)
        return cv2.warpPerspective(gray, M, (larg, alt)), M, True

    # sem papel identificável: trata a foto inteira como sendo a folha
    h, w = gray.shape
    src = np.array([[0, 0], [w, 0], [w, h], [0, h]], "float32")
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(gray, M, (larg, alt)), M, False


# --------------------------------------------------------------------------- #
# Etapa 2 — localizar as marcas fiduciais
# --------------------------------------------------------------------------- #
def template_marca(lado: int) -> np.ndarray:
    """Desenha a marca fiducial: cruz + quadrado com 2 quadrantes cinza.

    Geometria medida no PDF oficial (300 dpi, marca de 89 px): braço da cruz
    ocupa todo o lado, espessura ~6/89; quadrado central com lado 55/89 e borda
    5,5/89; quadrantes superior-esquerdo e inferior-direito em cinza ~147.
    """
    lado = max(9, int(lado))
    pad = max(2, int(round(lado * 0.16)))
    n = lado + 2 * pad
    if n % 2 == 0:
        n += 1
    img = np.full((n, n), 255, np.uint8)
    c = n // 2

    meio_q = int(round(lado * 27.5 / 89.0))
    x0, x1 = c - meio_q, c + meio_q
    img[x0:c, x0:c] = 147                       # quadrante superior-esquerdo
    img[c:x1 + 1, c:x1 + 1] = 147               # quadrante inferior-direito

    t_quad = max(1, int(round(lado * 5.5 / 89.0)))
    cv2.rectangle(img, (x0, x0), (x1, x1), 0, t_quad)

    t_cruz = max(1, int(round(lado * 6.0 / 89.0)))
    a0, a1 = c - lado // 2, c + lado // 2
    cv2.line(img, (a0, c), (a1, c), 0, t_cruz)
    cv2.line(img, (c, a0), (c, a1), 0, t_cruz)
    return img


def _pico_subpixel(mapa: np.ndarray, x: int, y: int) -> tuple[float, float]:
    """Refina o pico da correlação por parábola nos vizinhos imediatos."""
    dx = dy = 0.0
    h, w = mapa.shape
    if 0 < x < w - 1:
        a, b, c = float(mapa[y, x - 1]), float(mapa[y, x]), float(mapa[y, x + 1])
        den = a - 2 * b + c
        if abs(den) > 1e-9:
            dx = float(np.clip(0.5 * (a - c) / den, -1, 1))
    if 0 < y < h - 1:
        a, b, c = float(mapa[y - 1, x]), float(mapa[y, x]), float(mapa[y + 1, x])
        den = a - 2 * b + c
        if abs(den) > 1e-9:
            dy = float(np.clip(0.5 * (a - c) / den, -1, 1))
    return x + dx, y + dy


def _achar_marca(pagina: np.ndarray, alvo: tuple[float, float], lado_esp: float,
                 janela: float) -> tuple[float, float, float] | None:
    """Procura a marca perto de `alvo`. Devolve (x, y, score) ou None."""
    h, w = pagina.shape
    ax, ay = alvo
    x0 = int(max(0, ax - janela))
    y0 = int(max(0, ay - janela))
    x1 = int(min(w, ax + janela))
    y1 = int(min(h, ay + janela))
    roi = pagina[y0:y1, x0:x1]
    if roi.size == 0:
        return None

    melhor = None
    for fator in C.FIDUCIAL_ESCALAS:
        lado = int(round(lado_esp * fator))
        tpl = template_marca(lado)
        if tpl.shape[0] > roi.shape[0] or tpl.shape[1] > roi.shape[1]:
            continue
        mapa = cv2.matchTemplate(roi, tpl, cv2.TM_CCOEFF_NORMED)
        _, score, _, loc = cv2.minMaxLoc(mapa)
        if melhor is not None and score <= melhor[2]:
            continue
        px, py = _pico_subpixel(mapa, loc[0], loc[1])
        cx = x0 + px + (tpl.shape[1] - 1) / 2.0
        cy = y0 + py + (tpl.shape[0] - 1) / 2.0
        melhor = (cx, cy, float(score))

    if melhor is None or melhor[2] < C.FIDUCIAL_SCORE_MIN:
        return None
    return melhor


def _achar_fiduciais(pagina: np.ndarray) -> tuple[np.ndarray | None, list[float], str]:
    """Acha os 4 centros na página retificada. Ordem TL, TR, BR, BL."""
    h, w = pagina.shape
    fu, fv = T.FIDUCIAL_NA_PAGINA
    alvos = [(fu * w, fv * h), ((1 - fu) * w, fv * h),
             ((1 - fu) * w, (1 - fv) * h), (fu * w, (1 - fv) * h)]
    # o quadro fiducial ocupa (1 - 2*fu) da largura da página
    lado_esp = T.FIDUCIAL_MARCA_FRAC * (1 - 2 * fu) * w
    janela = C.FIDUCIAL_JANELA * w

    achados = [_achar_marca(pagina, a, lado_esp, janela) for a in alvos]
    ok = [i for i, m in enumerate(achados) if m is not None]
    scores = [m[2] if m else 0.0 for m in achados]

    if len(ok) == 4:
        return np.array([[m[0], m[1]] for m in achados], "float32"), scores, "fiduciais"

    if len(ok) == 3:
        # canto que falta = soma dos dois vizinhos menos o oposto (paralelogramo)
        falta = next(i for i in range(4) if achados[i] is None)
        viz = [(falta + 1) % 4, (falta + 3) % 4]
        opo = (falta + 2) % 4
        p = np.array([[m[0], m[1]] if m else [0.0, 0.0] for m in achados], "float32")
        p[falta] = p[viz[0]] + p[viz[1]] - p[opo]
        return p, scores, "fiduciais(3+1)"

    return None, scores, "papel"


def _quad_plausivel(pts: np.ndarray) -> bool:
    """Confere se os 4 centros formam um retângulo com o aspecto da folha."""
    if not cv2.isContourConvex(pts.reshape(-1, 1, 2).astype("float32")):
        return False
    topo = _lado(pts[0], pts[1])
    base = _lado(pts[3], pts[2])
    esq = _lado(pts[0], pts[3])
    dir_ = _lado(pts[1], pts[2])
    if min(topo, base, esq, dir_) < 1e-3:
        return False
    # lados opostos parecidos
    if abs(topo - base) / max(topo, base) > 0.25:
        return False
    if abs(esq - dir_) / max(esq, dir_) > 0.25:
        return False
    aspecto = ((esq + dir_) / 2.0) / ((topo + base) / 2.0)
    return abs(aspecto - T.ASPECTO) / T.ASPECTO <= C.FIDUCIAL_ASPECTO_TOL


# --------------------------------------------------------------------------- #
# API
# --------------------------------------------------------------------------- #
def registrar(image_bgr: np.ndarray, rotacao: int = 0) -> Registro:
    """Foto -> folha canônica retificada pelos 4 fiduciais.

    `rotacao` (0, 90, 180 ou 270, sentido horário) gira a foto antes de tudo;
    serve para reprocessar uma folha fotografada deitada ou de cabeça para
    baixo. Quem escolhe a rotação é `omr/flows.py`, pela cobertura da grade.
    """
    if image_bgr is None or getattr(image_bgr, "size", 0) == 0:
        raise EntradaInvalida("Imagem inválida ou não pôde ser decodificada.")
    if rotacao not in ROTACOES:
        raise ValueError(f"rotacao deve ser uma de {ROTACOES}")
    if image_bgr.ndim == 3:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = image_bgr
    altura_orig, largura_orig = gray.shape[:2]
    if rotacao:
        gray = cv2.rotate(gray, _CV_ROTACAO[rotacao])

    if min(gray.shape) < 200:
        raise EntradaInvalida(
            "Imagem pequena demais para leitura (mínimo ~200 px no menor lado). "
            "Fotografe a folha inteira com mais resolução."
        )

    pagina, M_pagina, achou_papel = _retificar_pagina(gray)
    pts_pagina, scores, origem = _achar_fiduciais(pagina)

    if pts_pagina is not None and not _quad_plausivel(pts_pagina):
        pts_pagina, origem = None, "papel"

    if pts_pagina is None:
        # fallback: usa a posição nominal das marcas dentro da página retificada
        if not achou_papel:
            raise OMRError(
                "Não encontrei os marcadores fiduciais nem as bordas da folha. "
                "Fotografe a folha inteira, reta, bem iluminada e sem cortar os "
                "quatro cantos."
            )
        h, w = pagina.shape
        fu, fv = T.FIDUCIAL_NA_PAGINA
        pts_pagina = np.array(
            [[fu * w, fv * h], [(1 - fu) * w, fv * h],
             [(1 - fu) * w, (1 - fv) * h], [fu * w, (1 - fv) * h]], "float32")
        origem = "papel"

    # leva os pontos da página de volta para a foto original e monta a
    # homografia direta foto -> canônica (uma única reamostragem)
    inv_pagina = np.linalg.inv(M_pagina)
    pts_orig = cv2.perspectiveTransform(
        pts_pagina.reshape(-1, 1, 2).astype("float32"), inv_pagina
    ).reshape(-1, 2).astype("float32")

    H = cv2.getPerspectiveTransform(pts_orig, CANTOS_CANONICOS)
    canonica = cv2.warpPerspective(
        gray, H, (CANON_IMG_W, CANON_IMG_H), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    if rotacao:
        # a homografia devolvida deve valer para a imagem ORIGINAL do usuário
        H = H @ _matriz_rotacao(rotacao, largura_orig, altura_orig)

    return Registro(
        canonica=canonica,
        homografia=H,
        origem_fiduciais=origem,
        scores=[round(s, 3) for s in scores],
        papel_detectado=achou_papel,
    )
