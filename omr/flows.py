"""
Os dois fluxos de leitura, um por página do modelo.

    ler_objetiva(foto)  -> página 1 (CARTÃO-RESPOSTA): número do aluno +
                           Linguagens 1..25 + Matemática 1..26
    ler_redacao(foto)   -> página 2 (PRODUÇÃO DE TEXTO): número do aluno +
                           quadro de correção do professor (situação + 5
                           competências)

Cada fluxo recebe a FOLHA INTEIRA fotografada; o recorte de cada área é feito
pelo template, ancorado nos marcadores fiduciais. Se a foto entregue for da
outra página, o fluxo recusa com uma mensagem clara em vez de devolver lixo.
"""
from __future__ import annotations

import cv2
import numpy as np

from . import config as C
from . import template as T
from .engine import LeituraFolha, ResultadoBloco, ResultadoCampo, ler_folha, sondar_objetiva
from .registration import (
    CANTOS_CANONICOS, ROTACOES, EntradaInvalida, OMRError, Registro, registrar,
)

__all__ = ["ler_objetiva", "ler_redacao", "ler_fluxo", "FLUXOS", "OMRError"]

# Cobertura = fração das bolhas do template encontradas na foto.
COBERTURA_ORIENTACAO = 0.55       # abaixo disso, procura outra orientação
COBERTURA_BOA = 0.85              # a partir daqui para de procurar
COBERTURA_MINIMA = 0.30           # abaixo disso, desiste: o registro não fecha


# --------------------------------------------------------------------------- #
# Registro + leitura, procurando a orientação certa
# --------------------------------------------------------------------------- #
def _tentar(image_bgr: np.ndarray, folha: T.Folha, rotacao: int):
    """Tenta uma orientação. Devolve (None, None) se ela não fechar.

    `EntradaInvalida` passa direto: se a imagem em si não serve, girar não vai
    ajudar e a mensagem específica é mais útil que a genérica do fim da busca.
    """
    try:
        reg = registrar(image_bgr, rotacao=rotacao)
        return reg, ler_folha(reg.canonica, folha)
    except EntradaInvalida:
        raise
    except OMRError:
        return None, None


def _registrar_e_ler(image_bgr: np.ndarray, folha: T.Folha) -> tuple[Registro, LeituraFolha, int]:
    """Lê a folha, descobrindo sozinho se a foto está deitada ou de ponta-cabeça.

    A orientação certa é a que reconhece mais bolhas do template — por isso a
    escolha é pela `cobertura`, e não por heurística de formato da imagem. O
    caminho normal (foto em pé) custa uma tentativa só: as outras três
    orientações só são testadas se a primeira não fechar.
    """
    reg, leitura = _tentar(image_bgr, folha, 0)
    rotacao = 0

    if leitura is None or leitura.cobertura < COBERTURA_ORIENTACAO:
        for outra in ROTACOES[1:]:
            reg2, leitura2 = _tentar(image_bgr, folha, outra)
            if leitura2 is None:
                continue
            if leitura is None or leitura2.cobertura > leitura.cobertura:
                reg, leitura, rotacao = reg2, leitura2, outra
            if leitura.cobertura >= COBERTURA_BOA:
                break

    if leitura is None:
        raise OMRError(
            "Não encontrei os marcadores fiduciais nem as bordas da folha em "
            "nenhuma orientação. Fotografe a folha inteira, reta, bem iluminada "
            "e sem cortar os quatro cantos."
        )

    if leitura.cobertura < COBERTURA_MINIMA:
        raise OMRError(
            "Não consegui alinhar a grade de bolhas da folha "
            f"(só {leitura.pares_globais} de {leitura.esperadas_globais} bolhas "
            "foram reconhecidas). Fotografe a folha inteira, sem dobras nem "
            "sombra forte, com os quatro marcadores dos cantos visíveis."
        )
    return reg, leitura, rotacao


def _conferir_pagina(canon: np.ndarray, leitura: LeituraFolha, folha: T.Folha) -> None:
    """Recusa a folha da outra página antes de devolver resultado nenhum."""
    if folha is T.FOLHA_OBJETIVA:
        bolhas = sum(
            leitura.blocos[b.nome].pares
            for b in (T.LINGUAGENS_B1, T.LINGUAGENS_B2, T.MATEMATICA_B1, T.MATEMATICA_B2)
            if b.nome in leitura.blocos
        )
        if bolhas < C.PAGINA_MIN_BOLHAS_OBJETIVA:
            raise OMRError(
                "Esta foto não parece o CARTÃO-RESPOSTA (não achei os blocos de "
                "Linguagens/Matemática). Se for a folha de PRODUÇÃO DE TEXTO, "
                "use o endpoint /anos-finais/omr/redacao."
            )
    else:
        if sondar_objetiva(canon) >= C.PAGINA_MIN_BOLHAS_OBJETIVA:
            raise OMRError(
                "Esta foto parece o CARTÃO-RESPOSTA (achei os blocos de "
                "Linguagens/Matemática). Para ler as respostas objetivas use o "
                "endpoint /anos-finais/omr/objetiva."
            )


# --------------------------------------------------------------------------- #
# Montagem das respostas
# --------------------------------------------------------------------------- #
def _campo_json(campo: ResultadoCampo) -> dict:
    return {
        "value": campo.valor,
        "status": campo.status,
        "marked": campo.marcadas,
        "fills": campo.fills,
    }


def _numero_aluno_json(bloco: ResultadoBloco) -> dict:
    """Monta o número do aluno a partir das 10 casas.

    `raw` sempre tem 10 caracteres: o dígito lido, `_` para casa em branco e `?`
    para casa ambígua/dupla. `value` só vem preenchido quando dá para afirmar o
    número: as casas em branco das PONTAS são ignoradas (número curto pode vir
    encostado à direita ou à esquerda) e o miolo precisa estar todo OK — buraco
    no meio ou casa ambígua derrubam para AMBIGUOUS em vez de chutar.
    """
    digitos = []
    bruto = []
    for i, campo in enumerate(bloco.campos, start=1):
        if campo.status == "OK":
            bruto.append(campo.valor)
        elif campo.status == "BLANK":
            bruto.append("_")
        else:
            bruto.append("?")
        digitos.append(
            {
                "position": i,
                "digit": campo.valor,
                "status": campo.status,
                "marked": campo.marcadas,
                "fills": campo.fills,
            }
        )

    raw = "".join(bruto)
    miolo = raw.strip("_")
    if not miolo:
        valor, status = None, "BLANK"
    elif all(ch.isdigit() for ch in miolo):
        valor, status = miolo, "OK"
    else:
        valor, status = None, "AMBIGUOUS"

    return {"value": valor, "raw": raw, "status": status, "digits": digitos}


def _questao_json(campo: ResultadoCampo) -> dict:
    return {
        "question": int(campo.chave),
        "answer": campo.valor,
        "status": campo.status,
        "marked": campo.marcadas,
        "fills": campo.fills,
    }


def _resumo(campos: list[ResultadoCampo]) -> dict:
    return {
        "ok": sum(c.status == "OK" for c in campos),
        "blank": sum(c.status == "BLANK" for c in campos),
        "multiple": sum(c.status == "MULTIPLE" for c in campos),
        "review": sum(c.status == "REVIEW" for c in campos),
    }


def _alinhamento_json(reg: Registro, leitura: LeituraFolha, rotacao: int) -> dict:
    return {
        "fiducials": reg.origem_fiduciais,
        "fiducial_scores": reg.scores,
        "paper_detected": reg.papel_detectado,
        "rotation": rotacao,
        "global_fit": leitura.ajuste_global,
        "coverage": round(leitura.cobertura, 3),
        "blocks": {
            nome: {"mode": b.alinhamento, "matched": b.pares, "expected": b.esperadas}
            for nome, b in leitura.blocos.items()
        },
    }


# --------------------------------------------------------------------------- #
# Fluxo 1 — objetiva
# --------------------------------------------------------------------------- #
def ler_objetiva(image_bgr: np.ndarray, debug: bool = False):
    """Lê a página 1: número do aluno + respostas de Linguagens e Matemática."""
    reg, leitura, rotacao = _registrar_e_ler(image_bgr, T.FOLHA_OBJETIVA)
    _conferir_pagina(reg.canonica, leitura, T.FOLHA_OBJETIVA)

    secoes = []
    todas: list[ResultadoCampo] = []
    for area, blocos in T.AREAS.items():
        campos: list[ResultadoCampo] = []
        for b in blocos:
            campos.extend(leitura.blocos[b.nome].campos)
        campos.sort(key=lambda c: int(c.chave))
        todas.extend(campos)
        secoes.append(
            {
                "name": area,
                "num_questions": len(campos),
                "summary": _resumo(campos),
                "results": [_questao_json(c) for c in campos],
            }
        )

    saida = {
        "flow": "objetiva",
        "student_number": _numero_aluno_json(leitura.blocos[T.BLOCO_NUMERO_ALUNO.nome]),
        "sections": secoes,
        "summary": _resumo(todas),
        "alignment": _alinhamento_json(reg, leitura, rotacao),
    }
    if debug:
        return saida, desenhar_debug(image_bgr, reg, leitura)
    return saida


# --------------------------------------------------------------------------- #
# Fluxo 2 — redação
# --------------------------------------------------------------------------- #
def ler_redacao(image_bgr: np.ndarray, debug: bool = False):
    """Lê a página 2: número do aluno + quadro de correção do professor."""
    reg, leitura, rotacao = _registrar_e_ler(image_bgr, T.FOLHA_REDACAO)
    _conferir_pagina(reg.canonica, leitura, T.FOLHA_REDACAO)

    por_chave: dict[str, ResultadoCampo] = {}
    for b in T.BLOCOS_CORRECAO:
        for campo in leitura.blocos[b.nome].campos:
            por_chave[campo.chave] = campo

    correcao = {}
    for chave in T.ORDEM_CORRECAO:
        campo = por_chave[chave]
        item = _campo_json(campo)
        if chave != "situacao":
            item["level"] = int(campo.valor) if campo.status == "OK" else None
        correcao[chave] = item

    campos = [por_chave[k] for k in T.ORDEM_CORRECAO]
    saida = {
        "flow": "redacao",
        "student_number": _numero_aluno_json(leitura.blocos[T.BLOCO_NUMERO_ALUNO.nome]),
        "correction": correcao,
        "summary": _resumo(campos),
        "alignment": _alinhamento_json(reg, leitura, rotacao),
    }
    if debug:
        return saida, desenhar_debug(image_bgr, reg, leitura)
    return saida


FLUXOS = {"objetiva": ler_objetiva, "redacao": ler_redacao}


def ler_fluxo(nome: str, image_bgr: np.ndarray, debug: bool = False):
    try:
        fn = FLUXOS[nome]
    except KeyError:
        raise ValueError(f"fluxo desconhecido: {nome!r} (use {sorted(FLUXOS)})") from None
    return fn(image_bgr, debug=debug)


# --------------------------------------------------------------------------- #
# Imagem anotada
# --------------------------------------------------------------------------- #
_COR_STATUS = {
    "OK": (0, 170, 0),
    "BLANK": (170, 170, 170),
    "MULTIPLE": (0, 0, 230),
    "REVIEW": (0, 140, 255),
}


def _poligono_original(reg: Registro, cx: float, cy: float, raio: float, lados: int = 16) -> np.ndarray:
    ang = np.linspace(0, 2 * np.pi, lados, endpoint=False)
    pts = np.stack([cx + raio * np.cos(ang), cy + raio * np.sin(ang)], axis=1)
    return reg.para_original(pts).round().astype(np.int32)


def desenhar_debug(image_bgr: np.ndarray, reg: Registro, leitura: LeituraFolha) -> np.ndarray:
    """Devolve a foto original com cada bolha lida marcada por cima.

    Verde cheio = considerada marcada; cinza = vazia; laranja = zona ambígua.
    Assim dá para ver, em uma olhada, se a grade caiu em cima das bolhas certas.
    """
    img = image_bgr.copy() if image_bgr.ndim == 3 else cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
    escala = max(1.0, min(img.shape[:2]) / 900.0)

    # moldura do quadro fiducial, para conferir o registro
    moldura = reg.para_original(CANTOS_CANONICOS).round().astype(np.int32)
    cv2.polylines(img, [moldura], True, (255, 0, 255), max(1, int(2 * escala)), cv2.LINE_AA)

    for bloco in leitura.blocos.values():
        tpl = next(b for b in T.FOLHAS["objetiva"].blocos + T.FOLHAS["redacao"].blocos
                   if b.nome == bloco.nome)
        for i, campo in enumerate(bloco.campos):
            rotulos = tpl.rotulos_de(i)
            for rot, (li, ci) in zip(rotulos, tpl.celulas_do_campo(i)):
                cx, cy = bloco.centros[li, ci]
                # a cor segue a DECISÃO do motor, não o limiar absoluto: uma
                # marca fraca escolhida pelo critério relativo tem que aparecer
                # verde, senão a imagem de debug contradiz o JSON
                if rot in campo.marcadas:
                    cor, esp = (0, 170, 0), max(2, int(3 * escala))
                elif campo.fills[rot] >= C.PISO_RELATIVO:
                    cor, esp = (0, 140, 255), max(2, int(3 * escala))
                else:
                    cor, esp = (150, 150, 150), max(1, int(1 * escala))
                poly = _poligono_original(reg, cx, cy, bloco.raio_px)
                cv2.polylines(img, [poly], True, cor, esp, cv2.LINE_AA)

            if campo.status in ("MULTIPLE", "REVIEW", "BLANK"):
                li, ci = tpl.celulas_do_campo(i)[0]
                cx, cy = bloco.centros[li, ci]
                p = reg.para_original(np.array([[cx - 2.2 * bloco.raio_px, cy]]))[0]
                cv2.putText(
                    img, campo.status[:4], (int(p[0]), int(p[1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45 * escala,
                    _COR_STATUS[campo.status], max(1, int(1.5 * escala)), cv2.LINE_AA,
                )
    return img
