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

from dataclasses import dataclass

from . import config as C
from . import template as T
from .engine import (
    LINHAS_MIN_OBJETIVA, LeituraFolha, ResultadoBloco, ResultadoCampo,
    identificar_modelo, ler_folha,
)
from .registration import (
    CANTOS_CANONICOS, ROTACOES, EntradaInvalida, OMRError, Registro, registrar,
)

__all__ = ["ler_objetiva", "ler_redacao", "ler_fluxo", "FLUXOS", "OMRError", "Contexto"]

# Cobertura = fração das bolhas do template encontradas na foto.
COBERTURA_ORIENTACAO = 0.55       # abaixo disso, procura outra orientação
COBERTURA_BOA = 0.85              # a partir daqui para de procurar
COBERTURA_MINIMA = 0.30           # abaixo disso, desiste: o registro não fecha


# --------------------------------------------------------------------------- #
# Registro + leitura, procurando a orientação certa
# --------------------------------------------------------------------------- #
@dataclass
class Contexto:
    """Tudo que a leitura de uma foto produziu, antes de virar JSON."""

    registro: Registro
    leitura: LeituraFolha
    rotacao: int
    modelo: T.Modelo
    folha: T.Folha
    diagnostico_modelo: dict


def _parece_outra_pagina(fluxo: str, linhas: int) -> bool:
    """As linhas de bolha vistas apontam para a OUTRA página deste modelo?

    A objetiva tem dezenas de linhas de questão; a de redação tem só as duas do
    quadro de correção. Quando o registro não fecha, essa contagem costuma ser
    a explicação real — e "você trocou o endpoint" ajuda muito mais do que
    "não consegui alinhar".
    """
    if fluxo == "redacao":
        return linhas >= LINHAS_MIN_OBJETIVA
    return 0 < linhas < LINHAS_MIN_OBJETIVA


def _erro_pagina_trocada(fluxo: str, linhas: int) -> OMRError:
    if fluxo == "objetiva":
        return OMRError(
            "Esta foto não parece um CARTÃO-RESPOSTA: não achei os blocos de "
            "questões de nenhum modelo (Anos Iniciais ou Anos Finais). Se for a "
            "folha de PRODUÇÃO DE TEXTO, use o endpoint /anos-iniciais/omr/redacao."
        )
    return OMRError(
        f"Esta foto parece um CARTÃO-RESPOSTA (achei {linhas} linhas de questões). "
        "Para ler as respostas objetivas use o endpoint /anos-iniciais/omr/objetiva."
    )


def _registrar_e_ler(image_bgr: np.ndarray, fluxo: str) -> Contexto:
    """Lê a foto descobrindo sozinha a orientação E o modelo de folha.

    Duas incógnitas são resolvidas antes de medir qualquer resposta:

    - **orientação**: a folha pode vir em pé, deitada ou de ponta-cabeça. A
      certa é a que reconhece mais bolhas do template (`cobertura`);
    - **modelo**: Anos Iniciais e Anos Finais são a mesma folha com contagens
      diferentes. Identificar errado NÃO falha ruidosamente — os dois têm o
      mesmo passo entre linhas, então o template errado devolve respostas
      deslocadas uma questão. Por isso o modelo é decidido pela geometria
      medida das linhas (`identificar_modelo`), que separa os dois por uma
      margem grande, e não pela cobertura, que os separa por pouco.

    O caminho normal (foto em pé) custa uma tentativa; as outras três
    orientações só são testadas se a primeira não fechar.
    """
    melhor: Contexto | None = None
    linhas_vistas = 0

    for rotacao in ROTACOES:
        try:
            reg = registrar(image_bgr, rotacao=rotacao)
        except EntradaInvalida:
            raise
        except OMRError:
            continue

        modelo, diag = identificar_modelo(reg.canonica, fluxo)
        linhas_vistas = max(linhas_vistas, diag["linhas_detectadas"])
        if modelo is None:
            continue

        folha = modelo.folhas[fluxo]
        try:
            leitura = ler_folha(reg.canonica, folha)
        except OMRError:
            continue

        if melhor is None or leitura.cobertura > melhor.leitura.cobertura:
            melhor = Contexto(reg, leitura, rotacao, modelo, folha, diag)
        if melhor.leitura.cobertura >= COBERTURA_BOA:
            break

    if melhor is None:
        # ou não achamos a folha, ou achamos e ela não é a página deste fluxo
        if linhas_vistas:
            raise _erro_pagina_trocada(fluxo, linhas_vistas)
        raise OMRError(
            "Não consegui identificar a folha em nenhuma orientação. Fotografe "
            "a folha inteira, reta, bem iluminada e com os quatro marcadores "
            "dos cantos visíveis."
        )

    if melhor.leitura.cobertura < COBERTURA_MINIMA:
        if _parece_outra_pagina(fluxo, linhas_vistas):
            raise _erro_pagina_trocada(fluxo, linhas_vistas)
        raise OMRError(
            "Não consegui alinhar a grade de bolhas da folha "
            f"(só {melhor.leitura.pares_globais} de "
            f"{melhor.leitura.esperadas_globais} bolhas foram reconhecidas). "
            "Fotografe a folha inteira, sem dobras nem sombra forte, com os "
            "quatro marcadores dos cantos visíveis."
        )
    return melhor


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


def _alinhamento_json(ctx: Contexto) -> dict:
    reg, leitura = ctx.registro, ctx.leitura
    return {
        "model_detection": ctx.diagnostico_modelo,
        "fiducials": reg.origem_fiduciais,
        "fiducial_scores": reg.scores,
        "paper_detected": reg.papel_detectado,
        "rotation": ctx.rotacao,
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
    """Lê a página objetiva: número do aluno + respostas por disciplina.

    Serve os dois modelos — Anos Iniciais (21 + 22 questões) e Anos Finais
    (25 + 26) — descobrindo qual é pela geometria da folha.
    """
    ctx = _registrar_e_ler(image_bgr, "objetiva")

    secoes = []
    todas: list[ResultadoCampo] = []
    for area, blocos in ctx.folha.areas.items():
        campos: list[ResultadoCampo] = []
        for b in blocos:
            campos.extend(ctx.leitura.blocos[b.nome].campos)
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
        "model": ctx.modelo.nome,
        "model_title": ctx.modelo.titulo,
        "student_number": _numero_aluno_json(ctx.leitura.blocos[T.BLOCO_NUMERO_ALUNO.nome]),
        "sections": secoes,
        "summary": _resumo(todas),
        "alignment": _alinhamento_json(ctx),
    }
    if debug:
        return saida, desenhar_debug(image_bgr, ctx)
    return saida


# --------------------------------------------------------------------------- #
# Fluxo 2 — redação (só existe no modelo Anos Finais)
# --------------------------------------------------------------------------- #
def ler_redacao(image_bgr: np.ndarray, debug: bool = False):
    """Lê a página de produção de texto: número do aluno + quadro de correção."""
    ctx = _registrar_e_ler(image_bgr, "redacao")

    por_chave: dict[str, ResultadoCampo] = {}
    for b in T.BLOCOS_CORRECAO:
        for campo in ctx.leitura.blocos[b.nome].campos:
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
        "model": ctx.modelo.nome,
        "model_title": ctx.modelo.titulo,
        "student_number": _numero_aluno_json(ctx.leitura.blocos[T.BLOCO_NUMERO_ALUNO.nome]),
        "correction": correcao,
        "summary": _resumo(campos),
        "alignment": _alinhamento_json(ctx),
    }
    if debug:
        return saida, desenhar_debug(image_bgr, ctx)
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


def desenhar_debug(image_bgr: np.ndarray, ctx: Contexto) -> np.ndarray:
    """Devolve a foto original com cada bolha lida marcada por cima.

    Verde cheio = considerada marcada; cinza = vazia; laranja = zona ambígua.
    Assim dá para ver, em uma olhada, se a grade caiu em cima das bolhas certas.
    """
    reg = ctx.registro
    # os blocos vêm da folha DETECTADA: os dois modelos usam os mesmos nomes de
    # bloco com contagens diferentes, então buscar por nome global daria o
    # template do modelo errado
    por_nome = {b.nome: b for b in ctx.folha.blocos}
    img = image_bgr.copy() if image_bgr.ndim == 3 else cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
    escala = max(1.0, min(img.shape[:2]) / 900.0)

    # moldura do quadro fiducial, para conferir o registro
    moldura = reg.para_original(CANTOS_CANONICOS).round().astype(np.int32)
    cv2.polylines(img, [moldura], True, (255, 0, 255), max(1, int(2 * escala)), cv2.LINE_AA)

    for bloco in ctx.leitura.blocos.values():
        tpl = por_nome[bloco.nome]
        for i, campo in enumerate(bloco.campos):
            rotulos = tpl.rotulos_de(i)
            for rot, (li, ci) in zip(rotulos, tpl.celulas_do_campo(i)):
                cx, cy = bloco.centros[li, ci]
                # a cor segue a DECISÃO do motor, não o limiar absoluto: marca
                # circulada aceita pela regra relativa tem que aparecer verde,
                # senão a imagem de debug contradiz o JSON
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
