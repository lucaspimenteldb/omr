"""
Testes da porta de entrada: bytes de arquivo -> matriz BGR.

O usuário fotografa com o celular e envia o que a câmera produziu. No iPhone
isso é HEIC, que o OpenCV não abre. O que estes testes garantem:

  - a mesma folha, no mesmo enquadramento, lê IGUAL em PNG, JPEG e HEIC —
    trocar de formato não pode mudar resposta nenhuma;
  - a orientação EXIF é aplicada igual nos dois caminhos de decodificação
    (OpenCV e Pillow), inclusive nas orientações espelhadas, que o laço de
    rotação do motor não conseguiria desfazer sozinho;
  - transparência vira papel branco, e não tinta;
  - arquivo que não é imagem devolve um recado que diz o que fazer.

Rode com:  .venv/bin/python -m pytest -q tests/test_entrada.py
"""
import io
import os
import sys

import cv2
import numpy as np
import pytest
from PIL import Image

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)

import make_test_images as G                              # noqa: E402
from omr import EntradaInvalida, decodificar, ler_arquivo, ler_fluxo  # noqa: E402
from omr import utils as U                                # noqa: E402


def _codificar(img_bgr: np.ndarray, formato: str, **kw) -> bytes:
    """BGR -> bytes no formato pedido, via Pillow (aceita HEIF)."""
    im = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    buf = io.BytesIO()
    im.save(buf, formato, **kw)
    return buf.getvalue()


@pytest.fixture(scope="module")
def folha():
    """Uma foto sintética de gabarito + o gabarito esperado."""
    jpeg, esperado = G.gerar_objetiva("facil", 1)
    return decodificar(jpeg), esperado


def _respostas(res: dict) -> list:
    """Só o que o cliente consome — sem os fills, que variam com a compressão."""
    return [(sec["name"], r["question"], r["answer"], r["status"])
            for sec in res["sections"] for r in sec["results"]]


# --------------------------------------------------------------------------- #
# HEIC do iPhone
# --------------------------------------------------------------------------- #
def test_heic_esta_disponivel():
    """Sem o plugin, todo iPhone recebe 422. É dependência de produção."""
    assert U.HEIF_DISPONIVEL, "instale pillow-heif (está em requirements.txt)"


def test_heic_decodifica_o_que_o_opencv_recusa(folha):
    img, _ = folha
    heic = _codificar(img, "HEIF", quality=90)
    assert heic[4:8] == b"ftyp", "o fixture não gerou um HEIC de verdade"
    assert cv2.imdecode(np.frombuffer(heic, np.uint8), cv2.IMREAD_COLOR) is None, \
        "o OpenCV passou a ler HEIC — o fallback do Pillow ficou sem cobertura aqui"

    saida = decodificar(heic)
    assert saida.shape == img.shape
    assert saida.dtype == np.uint8


@pytest.mark.parametrize("formato,kw", [
    ("PNG", {}),
    ("JPEG", {"quality": 92}),
    ("HEIF", {"quality": 90}),
    ("WEBP", {"quality": 92}),
])
def test_mesma_folha_le_igual_em_qualquer_formato(folha, formato, kw):
    """O formato do arquivo não pode mudar uma única resposta."""
    img, esperado = folha
    res = ler_fluxo("objetiva", decodificar(_codificar(img, formato, **kw)))

    assert res["student_number"]["value"] == esperado["student_number"]
    for sec in res["sections"]:
        for r in sec["results"]:
            alvo = esperado[sec["name"]][str(r["question"])]
            assert (r["answer"], r["status"]) == (alvo["answer"], alvo["status"]), \
                f"{formato} {sec['name']} Q{r['question']}"


def test_heic_e_png_dao_o_mesmo_resultado(folha):
    """Comparação direta entre os dois caminhos de decodificação."""
    img, _ = folha
    png = ler_fluxo("objetiva", decodificar(_codificar(img, "PNG")))
    heic = ler_fluxo("objetiva", decodificar(_codificar(img, "HEIF", quality=90)))
    assert _respostas(heic) == _respostas(png)


# --------------------------------------------------------------------------- #
# Orientação EXIF
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("orientacao", range(1, 9))
def test_exif_aplicado_igual_nos_dois_caminhos(orientacao):
    """O caminho do Pillow tem que girar como o cv2.imdecode gira.

    Cobre as 8 orientações, incluindo as 4 espelhadas (2, 4, 5, 7): o motor
    testa 0/90/180/270 sozinho, mas nenhuma rotação desfaz um espelho.
    """
    base = np.zeros((60, 100, 3), np.uint8)
    base[:20, :30] = (255, 0, 0)
    base[40:, 70:] = (0, 255, 0)
    im = Image.fromarray(base)

    exif = Image.Exif()
    exif[274] = orientacao
    buf = io.BytesIO()
    im.save(buf, "PNG", exif=exif)          # PNG com EXIF: o cv2 lê e gira
    via_cv2 = cv2.imdecode(np.frombuffer(buf.getvalue(), np.uint8), cv2.IMREAD_COLOR)
    via_pillow = U._para_bgr(Image.open(io.BytesIO(buf.getvalue())))

    assert via_pillow.shape == via_cv2.shape, f"orientação {orientacao}"
    assert np.array_equal(via_pillow, via_cv2), f"orientação {orientacao}"


def test_heic_de_iphone_nao_gira_duas_vezes():
    """O pillow-heif zera a tag EXIF porque o libheif já aplicou o `irot`.

    Medido em 82 fotos de iPhone: 59 trazem `original_orientation = 6` e já
    chegam em pé. Se alguém trocar o exif_transpose por esse campo, elas
    passam a sair deitadas — este teste quebra antes.
    """
    alto = np.zeros((80, 40, 3), np.uint8)
    alto[:20, :] = 255                       # topo branco: assimétrico no eixo y
    heic = _codificar(alto, "HEIF", quality=100)

    aberta = Image.open(io.BytesIO(heic))
    assert aberta.getexif().get(274, 1) == 1, "o plugin deveria ter zerado a tag"
    assert decodificar(heic).shape[:2] == (80, 40), "girou o que já estava em pé"


# --------------------------------------------------------------------------- #
# Modos de pixel que não são RGB de 8 bits
# --------------------------------------------------------------------------- #
def test_transparencia_vira_papel_branco():
    """RGBA achatado errado deixa o fundo preto — e preto o motor lê como tinta."""
    rgba = Image.new("RGBA", (30, 20), (0, 0, 0, 0))       # transparente puro
    buf = io.BytesIO()
    rgba.save(buf, "PNG")
    assert (decodificar(buf.getvalue()) == 255).all(), "fundo transparente não virou branco"


def test_cinza_de_16_bits_nao_estoura():
    """TIFF de scanner em 16 bits: converter direto para RGB trunca tudo."""
    a = np.linspace(0, 65535, 40 * 30, dtype=np.uint16).reshape(40, 30)
    buf = io.BytesIO()
    Image.frombytes("I;16", (30, 40), a.tobytes()).save(buf, "TIFF")

    saida = decodificar(buf.getvalue())
    assert saida.shape == (40, 30, 3) and saida.dtype == np.uint8
    assert saida.min() == 0 and saida.max() == 255            # faixa preservada
    assert 100 < int(saida.mean()) < 155                      # gradiente, não binário


# --------------------------------------------------------------------------- #
# Erros com recado útil
# --------------------------------------------------------------------------- #
def test_arquivo_vazio():
    with pytest.raises(EntradaInvalida, match="vazio"):
        decodificar(b"")


def test_pdf_explica_o_que_fazer():
    with pytest.raises(EntradaInvalida, match="PDF"):
        decodificar(b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n" + b"\x00" * 200)


def test_heic_truncado_diz_que_e_heic():
    with pytest.raises(EntradaInvalida, match="HEIC"):
        decodificar(b"\x00\x00\x00\x18ftypheic" + b"\x00" * 64)


def test_lixo_sugere_os_formatos():
    with pytest.raises(EntradaInvalida, match="JPG, PNG ou HEIC"):
        decodificar(b"nao sou uma imagem")


def test_arquivo_inexistente_cita_o_caminho(tmp_path):
    caminho = str(tmp_path / "nao_existe.heic")
    with pytest.raises(EntradaInvalida, match="nao_existe.heic"):
        ler_arquivo(caminho)


def test_ler_arquivo_abre_heic_do_disco(folha, tmp_path):
    """O CLI passa pelo mesmo caminho do upload."""
    img, _ = folha
    destino = tmp_path / "foto.heic"
    destino.write_bytes(_codificar(img, "HEIF", quality=90))
    assert ler_arquivo(str(destino)).shape == img.shape
