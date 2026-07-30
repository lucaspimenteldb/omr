"""
Leitor de gabaritos — modelo Cartão-Resposta Veloz (Anos Finais).

Dois fluxos, um por página da folha:

    ler_objetiva(foto)  -> nº do aluno + Linguagens (1..25) + Matemática (1..26)
    ler_redacao(foto)   -> nº do aluno + quadro de correção da redação

Ambos recebem a FOLHA INTEIRA fotografada e se ancoram nos 4 marcadores
fiduciais impressos nos cantos para corrigir a perspectiva.

Uso rápido:

    import cv2
    from omr import ler_objetiva
    resultado = ler_objetiva(cv2.imread("foto.jpg"))
"""
from .engine import LeituraFolha, ResultadoBloco, ResultadoCampo, ler_folha
from .flows import FLUXOS, desenhar_debug, ler_fluxo, ler_objetiva, ler_redacao
from .registration import EntradaInvalida, OMRError, registrar
from .utils import decodificar, ler_arquivo, ler_bytes

__all__ = [
    "OMRError",
    "EntradaInvalida",
    "ler_objetiva",
    "ler_redacao",
    "ler_fluxo",
    "FLUXOS",
    "ler_arquivo",
    "ler_bytes",
    "decodificar",
    "desenhar_debug",
    "registrar",
    "ler_folha",
    "LeituraFolha",
    "ResultadoBloco",
    "ResultadoCampo",
]
