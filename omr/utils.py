"""Entrada de imagem: bytes ou arquivo -> matriz BGR pronta para os fluxos."""
from __future__ import annotations

import cv2
import numpy as np

from .registration import EntradaInvalida, OMRError

# Fotos de celular passam fácil dos 12 MP; acima disso o custo cresce sem
# ganho de leitura nenhum (a folha canônica tem 1700 px de largura).
LADO_MAXIMO = 2600


def _reduzir(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    maior = max(h, w)
    if maior <= LADO_MAXIMO:
        return img
    esc = LADO_MAXIMO / float(maior)
    return cv2.resize(img, (int(round(w * esc)), int(round(h * esc))), interpolation=cv2.INTER_AREA)


def decodificar(image_bytes: bytes) -> np.ndarray:
    """Bytes de jpg/png -> BGR. Levanta OMRError se não decodificar."""
    if not image_bytes:
        raise EntradaInvalida("Arquivo vazio.")
    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise EntradaInvalida("Não consegui decodificar a imagem (formato não suportado?).")
    return _reduzir(img)


def ler_arquivo(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise EntradaInvalida(f"Não consegui abrir a imagem: {path}")
    return _reduzir(img)


def ler_bytes(image_bytes: bytes, fluxo: str, debug: bool = False):
    """Atalho: bytes -> resultado do fluxo pedido."""
    from .flows import ler_fluxo

    return ler_fluxo(fluxo, decodificar(image_bytes), debug=debug)
