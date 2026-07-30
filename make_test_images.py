#!/usr/bin/env python3
"""
Gera "fotos" sintéticas da folha para validar o leitor de ponta a ponta.

Sem um acervo de fotos reais anotadas, esta é a forma honesta de medir o motor:
partimos do PDF oficial renderizado, pintamos bolhas em posições que NÓS
escolhemos, e então degradamos a imagem como uma foto de celular degrada —
perspectiva, sombra, desfoque, ruído e compressão JPEG. Como o gabarito é
conhecido por construção, o teste vira uma comparação exata.

    python make_test_images.py            # gera tudo em test_images/sinteticas/
    python make_test_images.py --listar   # só mostra o que seria gerado

Sai também um `gabaritos.json` com a resposta certa de cada arquivo. Os testes
(`tests/test_leitura.py`) NÃO dependem desses arquivos: eles chamam as funções
daqui e geram as fotos em memória, com as mesmas sementes. O que fica em disco
serve para conferir na mão, no Postman ou com `test_cli.py --debug`.
"""
from __future__ import annotations

import argparse
import json
import os

import cv2
import numpy as np

from omr import template as T

RAIZ = os.path.dirname(os.path.abspath(__file__))
MODELO = os.path.join(RAIZ, "samples", "modelo")
SAIDA = os.path.join(RAIZ, "test_images", "sinteticas")

# Quadro fiducial no render 300 dpi (medido; ver tools/calibrar_template.py).
FID_X0, FID_Y0 = 82.0, 82.05
FID_W, FID_H = T.FIDUCIAL_W_PX, T.FIDUCIAL_H_PX


def _para_render(u: float, v: float) -> tuple[float, float]:
    return (FID_X0 + u * FID_W, FID_Y0 + v * FID_H)


# --------------------------------------------------------------------------- #
# Pintar as bolhas
# --------------------------------------------------------------------------- #
def pintar_bolha(img: np.ndarray, u: float, v: float, raio_norm: float,
                 rng: np.random.Generator, forca: float = 1.0) -> None:
    """Pinta uma bolha como caneta pinta: contorno irregular e centro deslocado.

    `forca` em 1.0 = bolha bem coberta; valores menores deixam a marca fraca
    (útil para exercitar a zona REVIEW).
    """
    cx, cy = _para_render(u, v)
    raio = raio_norm * FID_W
    cx += rng.normal(0, 0.07 * raio)
    cy += rng.normal(0, 0.07 * raio)

    lados = 24
    ang = np.linspace(0, 2 * np.pi, lados, endpoint=False)
    base = raio * (0.86 + 0.30 * forca)
    ruido = rng.normal(1.0, 0.06, lados)
    ruido = np.convolve(np.r_[ruido, ruido[:4]], np.ones(5) / 5, "valid")  # suaviza
    rr = base * ruido[:lados]
    pts = np.stack([cx + rr * np.cos(ang), cy + rr * np.sin(ang)], axis=1)

    tinta = int(rng.integers(25, 55))
    camada = img.copy()
    cv2.fillPoly(camada, [pts.round().astype(np.int32)], (tinta, tinta, tinta), cv2.LINE_AA)
    alpha = 0.55 + 0.45 * forca
    cv2.addWeighted(camada, alpha, img, 1 - alpha, 0, dst=img)


def marcar_campo(img: np.ndarray, bloco: T.Bloco, indice_campo: int, rotulo: str,
                 rng: np.random.Generator, forca: float = 1.0) -> None:
    rotulos = bloco.rotulos_de(indice_campo)
    celulas = bloco.celulas_do_campo(indice_campo)
    li, ci = celulas[rotulos.index(rotulo)]
    u, v = bloco.grade.centro(li, ci)
    pintar_bolha(img, u, v, bloco.grade.raio, rng, forca)


# --------------------------------------------------------------------------- #
# Degradar em "foto"
# --------------------------------------------------------------------------- #
def simular_foto(folha: np.ndarray, rng: np.random.Generator, *, perspectiva: float,
                 sombra: float, desfoque: int, ruido: float, qualidade: int,
                 largura: int = 1800) -> bytes:
    """Folha limpa -> bytes de um JPEG parecido com foto de celular."""
    h, w = folha.shape[:2]
    esc = largura / float(w)
    folha = cv2.resize(folha, (largura, int(round(h * esc))), interpolation=cv2.INTER_AREA)
    h, w = folha.shape[:2]

    # mesa em volta do papel, para a etapa de detecção do papel ter o que achar
    margem = int(0.09 * w)
    fundo_tom = int(rng.integers(70, 135))
    cena = np.full((h + 2 * margem, w + 2 * margem, 3), fundo_tom, np.float32)
    cena += rng.normal(0, 6, cena.shape)
    cena = np.clip(cena, 0, 255).astype(np.uint8)
    H, W = cena.shape[:2]

    # papel jogado na cena com perspectiva
    cantos_src = np.array([[0, 0], [w, 0], [w, h], [0, h]], "float32")
    alvo = np.array(
        [[margem, margem], [margem + w, margem], [margem + w, margem + h], [margem, margem + h]],
        "float32",
    )
    jitter = rng.uniform(-perspectiva, perspectiva, (4, 2)) * np.array([w, h])
    cantos_dst = (alvo + jitter).astype("float32")
    M = cv2.getPerspectiveTransform(cantos_src, cantos_dst)
    papel = cv2.warpPerspective(folha, M, (W, H), borderValue=(255, 255, 255))
    mascara = cv2.warpPerspective(np.full((h, w), 255, np.uint8), M, (W, H))
    cena = np.where(mascara[..., None] > 0, papel, cena)

    # iluminação irregular: um gradiente linear + um vinhetamento suave
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    ang = rng.uniform(0, 2 * np.pi)
    rampa = (np.cos(ang) * xx / W + np.sin(ang) * yy / H)
    rampa = (rampa - rampa.min()) / (rampa.max() - rampa.min() + 1e-9)
    luz = 1.0 - sombra * rampa
    r = np.hypot((xx - W / 2) / (W / 2), (yy - H / 2) / (H / 2))
    luz *= 1.0 - 0.18 * sombra * np.clip(r, 0, 1.4) ** 2
    cena = np.clip(cena.astype(np.float32) * luz[..., None], 0, 255).astype(np.uint8)

    if desfoque > 0:
        k = int(desfoque) * 2 + 1
        cena = cv2.GaussianBlur(cena, (k, k), 0)
    if ruido > 0:
        cena = np.clip(cena.astype(np.float32) + rng.normal(0, ruido, cena.shape), 0, 255).astype(np.uint8)

    ok, buf = cv2.imencode(".jpg", cena, [cv2.IMWRITE_JPEG_QUALITY, qualidade])
    if not ok:
        raise RuntimeError("falha ao codificar JPEG")
    return buf.tobytes()


# --------------------------------------------------------------------------- #
# Os casos de teste
# --------------------------------------------------------------------------- #
def _numero(img, numero: str, rng, forca: float = 1.0) -> str:
    """Pinta o número do aluno alinhado à direita, como o aluno preenche."""
    bloco = T.BLOCO_NUMERO_ALUNO
    inicio = T.N_DIGITOS - len(numero)
    for i, d in enumerate(numero):
        marcar_campo(img, bloco, inicio + i, d, rng, forca)
    return numero


CENARIOS = {
    "facil": dict(perspectiva=0.006, sombra=0.10, desfoque=0, ruido=2.0, qualidade=95),
    "medio": dict(perspectiva=0.020, sombra=0.26, desfoque=1, ruido=4.0, qualidade=85),
    "dificil": dict(perspectiva=0.038, sombra=0.42, desfoque=2, ruido=7.0, qualidade=72),
}


def _abrir_modelo(pagina: int) -> np.ndarray:
    caminho = os.path.join(MODELO, f"pagina{pagina}_300dpi.png")
    img = cv2.imread(caminho)
    if img is None:
        raise SystemExit(
            f"{caminho} não encontrado — rode `python tools/calibrar_template.py --render`"
        )
    return img


def gerar_objetiva(cenario: str, semente: int) -> tuple[bytes, dict]:
    rng = np.random.default_rng(semente)
    img = _abrir_modelo(1)

    alt = "ABCD"
    gabarito = {"student_number": _numero(img, "1207", rng),
                "linguagens": {}, "matematica": {}}

    especiais = {("linguagens", 7): "BLANK", ("linguagens", 20): "MULTIPLE",
                 ("matematica", 3): "MULTIPLE", ("matematica", 26): "BLANK"}
    for area, blocos in T.AREAS.items():
        for bloco in blocos:
            for i, chave in enumerate(bloco.chaves):
                q = int(chave)
                caso = especiais.get((area, q))
                if caso == "BLANK":
                    gabarito[area][chave] = {"status": "BLANK", "answer": None}
                    continue
                letra = alt[(q * 7 + (0 if area == "linguagens" else 3)) % 4]
                marcar_campo(img, bloco, i, letra, rng)
                if caso == "MULTIPLE":
                    outra = alt[(alt.index(letra) + 2) % 4]
                    marcar_campo(img, bloco, i, outra, rng)
                    gabarito[area][chave] = {"status": "MULTIPLE", "answer": None}
                else:
                    gabarito[area][chave] = {"status": "OK", "answer": letra}

    return simular_foto(img, rng, **CENARIOS[cenario]), gabarito


def gerar_redacao(cenario: str, semente: int) -> tuple[bytes, dict]:
    rng = np.random.default_rng(semente)
    img = _abrir_modelo(2)

    gabarito = {"student_number": _numero(img, "1207", rng), "correction": {}}
    escolhas = {"situacao": "C", "competencia_01": "3", "competencia_02": "4",
                "competencia_03": "0", "competencia_04": "2", "competencia_05": "1"}
    duplicar = {"competencia_04"}          # exercita o caso MULTIPLE
    vazios = {"competencia_05"}            # exercita o caso BLANK

    for bloco in T.BLOCOS_CORRECAO:
        for i, chave in enumerate(bloco.chaves):
            if chave in vazios:
                gabarito["correction"][chave] = {"status": "BLANK", "value": None}
                continue
            rot = escolhas[chave]
            marcar_campo(img, bloco, i, rot, rng)
            if chave in duplicar:
                rotulos = bloco.rotulos_de(i)
                outro = rotulos[(rotulos.index(rot) + 2) % len(rotulos)]
                marcar_campo(img, bloco, i, outro, rng)
                gabarito["correction"][chave] = {"status": "MULTIPLE", "value": None}
            else:
                gabarito["correction"][chave] = {"status": "OK", "value": rot}

    return simular_foto(img, rng, **CENARIOS[cenario]), gabarito


CASOS = [
    ("objetiva_facil.jpg", "objetiva", "facil", 1),
    ("objetiva_medio.jpg", "objetiva", "medio", 2),
    ("objetiva_dificil.jpg", "objetiva", "dificil", 3),
    ("redacao_facil.jpg", "redacao", "facil", 11),
    ("redacao_medio.jpg", "redacao", "medio", 12),
    ("redacao_dificil.jpg", "redacao", "dificil", 13),
]


def main() -> None:
    ap = argparse.ArgumentParser(description="Gera fotos sintéticas de teste.")
    ap.add_argument("--listar", action="store_true", help="só lista os casos")
    args = ap.parse_args()

    if args.listar:
        for nome, fluxo, cenario, _ in CASOS:
            print(f"{nome:24s} fluxo={fluxo:9s} cenário={cenario}")
        return

    os.makedirs(SAIDA, exist_ok=True)
    gabaritos = {}
    for nome, fluxo, cenario, semente in CASOS:
        gerar = gerar_objetiva if fluxo == "objetiva" else gerar_redacao
        jpeg, gab = gerar(cenario, semente)
        destino = os.path.join(SAIDA, nome)
        with open(destino, "wb") as fh:
            fh.write(jpeg)
        gabaritos[nome] = {"fluxo": fluxo, "cenario": cenario, "gabarito": gab}
        print(f"{destino}  ({len(jpeg) / 1024:.0f} KB, {cenario})")

    with open(os.path.join(SAIDA, "gabaritos.json"), "w", encoding="utf-8") as fh:
        json.dump(gabaritos, fh, ensure_ascii=False, indent=2)
    print(os.path.join(SAIDA, "gabaritos.json"))


if __name__ == "__main__":
    main()
