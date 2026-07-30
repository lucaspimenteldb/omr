"""
Motor de OMR (Optical Mark Recognition).

Pipeline:
  1. detecta os 2 boxes de coluna na foto (contornos retangulares grandes);
  2. retifica (perspective warp) cada box para um retângulo canônico;
  3. binariza cada box (Otsu global);
  4. amostra o percentual de preenchimento de cada bolha usando a grade fixa
     do template (config.py);
  5. classifica cada questão em: OK / BLANK / MULTIPLE / REVIEW.

O foco é a precisão nos três cenários:
  - questão SEM alternativa marcada        -> status "BLANK"
  - questão com MAIS DE UMA marcada         -> status "MULTIPLE"
  - questão com EXATAMENTE UMA marcada      -> status "OK" (answer = letra)
  - qualquer bolha em zona ambígua          -> status "REVIEW" (checagem manual)
"""
from __future__ import annotations

import cv2
import numpy as np

from . import config as C


class OMRError(Exception):
    """Falha estrutural: não foi possível ler a folha (boxes não encontrados etc.)."""


# --------------------------------------------------------------------------- #
# Detecção e retificação dos boxes de coluna
# --------------------------------------------------------------------------- #
def _order_corners(pts: np.ndarray) -> np.ndarray:
    """Ordena 4 pontos como [top-left, top-right, bottom-right, bottom-left]."""
    pts = pts.reshape(4, 2).astype("float32")
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array(
        [pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]],
        dtype="float32",
    )


def _find_boxes(gray: np.ndarray) -> list[np.ndarray]:
    """Encontra os boxes de coluna e devolve seus 4 cantos, ordenados esq->dir."""
    h, w = gray.shape
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    th = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 10
    )
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    area_total = float(h * w)
    boxes = []
    for c in cnts:
        area = cv2.contourArea(c)
        if not (C.BOX_MIN_AREA_FRAC * area_total <= area <= C.BOX_MAX_AREA_FRAC * area_total):
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        x, y, bw, bh = cv2.boundingRect(approx)
        if bh < bw:  # box de coluna é "em pé" (mais alto que largo)
            continue
        boxes.append((area, _order_corners(approx)))

    # pega os N maiores e ordena da esquerda para a direita
    boxes.sort(key=lambda b: b[0], reverse=True)
    boxes = [b[1] for b in boxes[: C.NUM_BOXES]]
    if len(boxes) < C.NUM_BOXES:
        raise OMRError(
            f"Esperava {C.NUM_BOXES} colunas de respostas, encontrei {len(boxes)}. "
            "Verifique se a folha inteira está enquadrada, reta e com bom contraste."
        )
    boxes.sort(key=lambda q: q[0][0])  # por X do canto superior-esquerdo
    return boxes


def _warp(gray: np.ndarray, quad: np.ndarray) -> np.ndarray:
    dst = np.array([[0, 0], [C.GRID_W, 0], [C.GRID_W, C.GRID_H], [0, C.GRID_H]], "float32")
    M = cv2.getPerspectiveTransform(quad, dst)
    return cv2.warpPerspective(gray, M, (C.GRID_W, C.GRID_H))


# --------------------------------------------------------------------------- #
# Detecção da grade de bolhas (posições REAIS, não fixas)
# --------------------------------------------------------------------------- #
def _normalize_illumination(warp: np.ndarray) -> np.ndarray:
    """Flat-field: divide a imagem por uma versão muito borrada -> remove o
    gradiente de sombra (persiana, iluminação lateral)."""
    bg = cv2.GaussianBlur(warp, (0, 0), C.FLATFIELD_SIGMA)
    return cv2.divide(warp, bg, scale=255).astype("uint8")


def _split_k(vals, k):
    """Agrupa valores 1D em k grupos cortando nas (k-1) maiores lacunas.
    Devolve as medianas dos grupos, ordenadas; None se houver < k valores."""
    v = np.sort(np.asarray(vals, dtype=float))
    if len(v) < k:
        return None
    cut = np.sort(np.argsort(np.diff(v))[-(k - 1):])
    return np.array([np.median(g) for g in np.split(v, cut + 1)])


def _linfit(centers: np.ndarray) -> np.ndarray:
    """Regulariza para espaçamento uniforme: ajusta center[i] = a + b*i."""
    idx = np.arange(len(centers))
    b, a = np.polyfit(idx, centers, 1)
    return a + b * idx


def _detect_grid(warp: np.ndarray):
    """Detecta as posições reais das 4 colunas e 8 linhas de bolhas dentro do
    box já retificado. Devolve (cols[4], rows[8], raio_mediano).
    Levanta OMRError se a grade não puder ser recuperada com confiança."""
    norm = _normalize_illumination(warp)
    th = cv2.adaptiveThreshold(
        norm, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, C.ADAPT_BLOCK, C.ADAPT_C,
    )
    cnts, _ = cv2.findContours(th, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cand = []
    for c in cnts:
        a = cv2.contourArea(c)
        if a < 300:
            continue
        (cx, cy), r = cv2.minEnclosingCircle(c)
        if not (0.66 * C.BUBBLE_R_EXPECTED <= r <= 1.55 * C.BUBBLE_R_EXPECTED):
            continue
        if a / (np.pi * r * r) < 0.45:          # descarta formas não-circulares
            continue
        cand.append((cx, cy, r))
    if len(cand) < 20:
        raise OMRError(
            "Poucas bolhas detectadas em uma coluna; foto possivelmente fora de "
            "foco, cortada ou com contraste ruim. Refaça a foto mais reta e nítida."
        )
    colc = _split_k([p[0] for p in cand], 4)
    rowc = _split_k([p[1] for p in cand], 8)
    if colc is None or rowc is None:
        raise OMRError("Não foi possível separar 4 colunas / 8 linhas de bolhas.")
    cols, rows = _linfit(colc), _linfit(rowc)
    col_pitch = float(np.mean(np.diff(cols)))
    row_pitch = float(np.mean(np.diff(rows)))
    if col_pitch <= 0 or row_pitch <= 0:
        raise OMRError("Grade degenerada (espaçamento inválido).")
    if (np.max(np.abs(colc - cols)) > 0.30 * col_pitch or
            np.max(np.abs(rowc - rows)) > 0.30 * row_pitch):
        raise OMRError(
            "Grade irregular; alinhamento não confiável. Refaça a foto mais reta."
        )
    rmed = float(np.median([p[2] for p in cand]))
    return cols, rows, rmed


def _measure_fills(warp: np.ndarray, cols, rows, rmed: float):
    """% de preenchimento de cada bolha nas posições DETECTADAS. Devolve uma
    lista de 8 linhas, cada uma com 4 valores (A, B, C, D)."""
    binimg = cv2.threshold(
        _normalize_illumination(warp), 0, 255,
        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU,
    )[1]
    rs = int(round(C.SAMPLE_R_FRAC * rmed))
    mask = np.zeros((2 * rs, 2 * rs), np.uint8)
    cv2.circle(mask, (rs, rs), rs, 255, -1)
    area = cv2.countNonZero(mask)
    grid = []
    for ry in rows:
        row = []
        for cx in cols:
            x0, y0 = int(cx - rs), int(ry - rs)
            patch = binimg[y0 : y0 + 2 * rs, x0 : x0 + 2 * rs]
            if patch.shape[:2] != (2 * rs, 2 * rs):
                row.append(0.0)
            else:
                dark = cv2.countNonZero(cv2.bitwise_and(patch, patch, mask=mask))
                row.append(100.0 * dark / area if area else 0.0)
        grid.append(row)
    return grid


def _classify(fills: dict[str, float]) -> tuple[str | None, str]:
    """Decide (answer, status) para uma questão a partir dos 4 preenchimentos."""
    marked = [ch for ch, f in fills.items() if f >= C.MARK_THRESHOLD]
    uncertain = [ch for ch, f in fills.items() if C.REVIEW_LOW <= f < C.MARK_THRESHOLD]

    if len(marked) >= 2:
        return None, "MULTIPLE"
    if uncertain:                       # algo entre "vazio" e "marcado": humano decide
        return (marked[0] if marked else None), "REVIEW"
    if len(marked) == 1:
        return marked[0], "OK"
    return None, "BLANK"


# --------------------------------------------------------------------------- #
# API pública
# --------------------------------------------------------------------------- #
def process_image(image_bgr: np.ndarray, draw_debug: bool = False) -> dict:
    """
    Lê uma foto do gabarito e devolve as marcações.

    Retorno:
      {
        "num_questions": int,
        "results": [
          {"question": 1, "answer": "D", "status": "OK",
           "marked": ["D"], "fills": {"A":27.1,"B":30.3,"C":18.7,"D":100.0}},
          ...
        ],
        "summary": {"ok":.., "blank":.., "multiple":.., "review":..},
      }
    Levanta OMRError se a folha não puder ser localizada.
    """
    if image_bgr is None:
        raise OMRError("Imagem inválida / não pôde ser decodificada.")
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    boxes = _find_boxes(gray)

    debug_img = image_bgr.copy() if draw_debug else None
    results = []
    q = 1
    for quad in boxes:
        warp = _warp(gray, quad)
        cols, rows, rmed = _detect_grid(warp)      # posições REAIS das bolhas
        grid = _measure_fills(warp, cols, rows, rmed)

        # matriz inversa p/ desenhar debug de volta na imagem original
        inv_M = None
        if draw_debug:
            dst = np.array([[0, 0], [C.GRID_W, 0], [C.GRID_W, C.GRID_H], [0, C.GRID_H]], "float32")
            inv_M = cv2.getPerspectiveTransform(dst, quad)

        for ri, ry in enumerate(rows):
            fills = {ch: round(grid[ri][ci], 1) for ci, ch in enumerate(C.CHOICES)}
            answer, status = _classify(fills)
            results.append(
                {
                    "question": q,
                    "answer": answer,
                    "status": status,
                    "marked": [ch for ch, f in fills.items() if f >= C.MARK_THRESHOLD],
                    "fills": fills,
                }
            )
            if draw_debug:
                _draw_row(debug_img, inv_M, cols, ry, rmed, fills, status)
            q += 1

    summary = {
        "ok": sum(r["status"] == "OK" for r in results),
        "blank": sum(r["status"] == "BLANK" for r in results),
        "multiple": sum(r["status"] == "MULTIPLE" for r in results),
        "review": sum(r["status"] == "REVIEW" for r in results),
    }
    out = {"num_questions": len(results), "results": results, "summary": summary}
    if draw_debug:
        return out, debug_img
    return out


def _draw_row(img, inv_M, cols, ry, rmed, fills, status):
    """Marca cada bolha na imagem original, nas posições DETECTADAS
    (verde=marcada, cinza=vazia)."""
    color_status = {
        "OK": (0, 170, 0), "BLANK": (0, 200, 200),
        "MULTIPLE": (0, 0, 230), "REVIEW": (0, 140, 255),
    }[status]
    radius = max(3, int(rmed * 1.2))
    for ch, cx in zip(C.CHOICES, cols):
        pt = cv2.perspectiveTransform(np.array([[[cx, ry]]], "float32"), inv_M)[0][0]
        marked = fills[ch] >= C.MARK_THRESHOLD
        col = (0, 170, 0) if marked else (150, 150, 150)
        cv2.circle(img, (int(pt[0]), int(pt[1])), radius, col, 4 if marked else 2)
    # tag de status à esquerda da 1ª bolha da linha
    p0 = cv2.perspectiveTransform(np.array([[[cols[0] - 2 * rmed, ry]]], "float32"), inv_M)[0][0]
    cv2.putText(img, status[:4], (int(p0[0]), int(p0[1])),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color_status, 2, cv2.LINE_AA)


def read_image_file(path: str, draw_debug: bool = False):
    img = cv2.imread(path)
    return process_image(img, draw_debug=draw_debug)


def decode_and_process(image_bytes: bytes, draw_debug: bool = False):
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return process_image(img, draw_debug=draw_debug)
