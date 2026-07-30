"""
Gera imagens de teste a partir do gabarito de referência para exercitar os
3 cenários: EM BRANCO e MÚLTIPLA (o de marcação ÚNICA é a própria referência).

- test_blank.png    : apaga a marca da Q1 (D) -> Q1 deve virar BLANK
- test_multiple.png : adiciona 2ª marca na Q2 (A, além do B) -> Q2 deve virar MULTIPLE
- test_mixed.png    : as duas modificações juntas
"""
import cv2
import numpy as np
from omr import engine, config as C

img = cv2.imread("samples/gabarito_matematica.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
boxes = engine._find_boxes(gray)

def center(box_idx, row_idx, col_idx):
    quad = boxes[box_idx]
    dst = np.array([[0, 0], [C.CANON_W, 0], [C.CANON_W, C.CANON_H], [0, C.CANON_H]], "float32")
    inv = cv2.getPerspectiveTransform(dst, quad)
    pt = cv2.perspectiveTransform(
        np.array([[[C.COL_X[col_idx], C.ROW_Y[row_idx]]]], "float32"), inv)[0][0]
    return int(pt[0]), int(pt[1])

def erase(im, box, row, col):   # pinta de branco -> vira bolha vazia
    cv2.circle(im, center(box, row, col), 14, (255, 255, 255), -1)
def fill(im, box, row, col):    # pinta de escuro -> vira bolha marcada
    cv2.circle(im, center(box, row, col), 11, (70, 40, 30), -1)

# BLANK: apaga Q1 (box0, row0) alternativa D (col3)
blank = img.copy(); erase(blank, 0, 0, 3)
cv2.imwrite("test_images/test_blank.png", blank)

# MULTIPLE: Q2 (box0, row1) já tem B(col1); adiciona A(col0)
mult = img.copy(); fill(mult, 0, 1, 0)
cv2.imwrite("test_images/test_multiple.png", mult)

# MIXED
mix = img.copy(); erase(mix, 0, 0, 3); fill(mix, 0, 1, 0)
cv2.imwrite("test_images/test_mixed.png", mix)

print("Imagens de teste geradas em test_images/")
