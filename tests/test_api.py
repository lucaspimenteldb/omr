"""
Testes da API — o contrato que o cliente (app / Postman) enxerga.

Checa o caminho feliz dos dois endpoints, o formato do JSON, o endpoint de
debug e os erros: arquivo vazio, imagem inválida e folha da página errada.
"""
import os
import sys

import pytest

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)

pytest.importorskip("httpx", reason="TestClient do FastAPI precisa de httpx")

from fastapi.testclient import TestClient   # noqa: E402

import make_test_images as G                # noqa: E402
from app import app                         # noqa: E402
from omr import template as T               # noqa: E402

cliente = TestClient(app)


@pytest.fixture(scope="module")
def jpegs():
    obj, gab_obj = G.gerar_objetiva("facil", 1)
    red, gab_red = G.gerar_redacao("facil", 11)
    return {"objetiva": (obj, gab_obj), "redacao": (red, gab_red)}


def _post(rota, jpeg, nome="folha.jpg"):
    return cliente.post(rota, files={"file": (nome, jpeg, "image/jpeg")})


def test_health():
    r = cliente.get("/health")
    assert r.status_code == 200
    corpo = r.json()
    assert corpo["status"] == "ok"
    assert corpo["fluxos"]["objetiva"]["linguagens"] == 25
    assert corpo["fluxos"]["objetiva"]["matematica"] == 26
    assert corpo["fluxos"]["redacao"]["campos"] == list(T.ORDEM_CORRECAO)


def test_objetiva_contrato(jpegs):
    jpeg, gab = jpegs["objetiva"]
    r = _post("/anos-finais/omr/objetiva", jpeg)
    assert r.status_code == 200, r.text
    corpo = r.json()

    assert corpo["filename"] == "folha.jpg"
    assert corpo["flow"] == "objetiva"
    assert corpo["student_number"]["value"] == gab["student_number"]
    assert {s["name"] for s in corpo["sections"]} == {"linguagens", "matematica"}
    assert set(corpo["summary"]) == {"ok", "blank", "multiple", "review"}
    assert sum(corpo["summary"].values()) == 51        # 25 + 26 questões

    q1 = corpo["sections"][0]["results"][0]
    assert set(q1) == {"question", "answer", "status", "marked", "fills"}
    assert set(q1["fills"]) == set(T.ALTERNATIVAS)
    assert corpo["alignment"]["fiducials"] == "fiduciais"


def test_redacao_contrato(jpegs):
    jpeg, gab = jpegs["redacao"]
    r = _post("/anos-finais/omr/redacao", jpeg, nome="redacao.jpg")
    assert r.status_code == 200, r.text
    corpo = r.json()

    assert corpo["flow"] == "redacao"
    assert corpo["student_number"]["value"] == gab["student_number"]
    assert list(corpo["correction"]) == list(T.ORDEM_CORRECAO)

    situacao = corpo["correction"]["situacao"]
    assert set(situacao["fills"]) == set(T.SITUACOES)
    assert "level" not in situacao                       # situação é letra, não nível

    c1 = corpo["correction"]["competencia_01"]
    assert set(c1["fills"]) == set(T.NIVEIS)
    assert c1["level"] == int(c1["value"])

    for chave, esperado in gab["correction"].items():
        assert corpo["correction"][chave]["status"] == esperado["status"]
        assert corpo["correction"][chave]["value"] == esperado["value"]


@pytest.mark.parametrize(
    "rota", ["/anos-finais/omr/objetiva/debug", "/anos-finais/omr/redacao/debug"])
def test_debug_devolve_png(jpegs, rota):
    fluxo = "objetiva" if "objetiva" in rota else "redacao"
    r = _post(rota, jpegs[fluxo][0])
    assert r.status_code == 200, r.text
    assert r.headers["content-type"] == "image/png"
    assert r.content[:8] == b"\x89PNG\r\n\x1a\n"


def test_folha_trocada_devolve_422_com_dica(jpegs):
    r = _post("/anos-finais/omr/redacao", jpegs["objetiva"][0])
    assert r.status_code == 422
    assert "/anos-finais/omr/objetiva" in r.json()["detail"]

    r = _post("/anos-finais/omr/objetiva", jpegs["redacao"][0])
    assert r.status_code == 422
    assert "/anos-finais/omr/redacao" in r.json()["detail"]


def test_upload_heic_do_iphone(jpegs):
    """O celular manda HEIC; o endpoint tem que aceitar como aceita JPEG."""
    import io

    import cv2
    import numpy as np
    from PIL import Image

    from omr import decodificar

    jpeg, gab = jpegs["objetiva"]
    img = decodificar(jpeg)
    buf = io.BytesIO()
    Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).save(buf, "HEIF", quality=90)
    heic = buf.getvalue()
    assert cv2.imdecode(np.frombuffer(heic, np.uint8), cv2.IMREAD_COLOR) is None

    r = cliente.post("/anos-finais/omr/objetiva",
                     files={"file": ("IMG_4821.HEIC", heic, "image/heic")})
    assert r.status_code == 200, r.text
    corpo = r.json()
    assert corpo["filename"] == "IMG_4821.HEIC"
    assert corpo["student_number"]["value"] == gab["student_number"]


def test_upload_malformado_explica_o_que_fazer(jpegs):
    """Corpo cru com Content-Type de multipart: o engano mais comum no Postman.

    O parser do Starlette devolve "Expected boundary character 45, got 148 at
    index 2", que não ajuda ninguém a montar a requisição direito.
    """
    r = cliente.post(
        "/anos-finais/omr/objetiva",
        content=jpegs["objetiva"][0],
        headers={"Content-Type": "multipart/form-data; boundary=------------------------12345"},
    )
    assert r.status_code == 400
    corpo = r.json()
    assert "form-data" in corpo["detail"] and "Content-Type" in corpo["detail"]
    assert corpo["parser"], "a mensagem crua do parser some do diagnóstico"

    # sem isto o diagnóstico vira adivinhação sobre o cliente
    recebido = corpo["recebido"]
    assert recebido["comeca_com_boundary"] is False
    assert recebido["primeiros_bytes_hex"], "o início do corpo não foi capturado"
    assert "JFIF" in recebido["primeiros_bytes_texto"], \
        "os bytes relatados não são os do arquivo que o cliente mandou cru"
    assert recebido["content_type"].startswith("multipart/form-data")


def test_multipart_sem_boundary():
    r = cliente.post("/anos-finais/omr/objetiva", content=b"--X\r\n\r\n",
                     headers={"Content-Type": "multipart/form-data"})
    assert r.status_code == 400
    assert "boundary" in r.json()["parser"].lower()


def test_pdf_recebe_recado_especifico():
    """PDF é o engano mais comum depois do HEIC — o 422 tem que dizer o que fazer."""
    r = _post("/anos-finais/omr/objetiva", b"%PDF-1.7\n" + b"\x00" * 300, nome="folha.pdf")
    assert r.status_code == 422
    assert "PDF" in r.json()["detail"]


@pytest.mark.parametrize("rota", ["/anos-finais/omr/objetiva", "/anos-finais/omr/redacao"])
def test_arquivo_vazio_e_invalido(rota):
    assert _post(rota, b"").status_code == 422
    assert _post(rota, b"nao sou uma imagem").status_code == 422


@pytest.mark.parametrize("rota", ["/anos-finais/omr/objetiva", "/anos-finais/omr/redacao"])
def test_sem_arquivo_da_422(rota):
    assert cliente.post(rota).status_code == 422


def test_rota_antiga_avisa_a_troca():
    r = cliente.post("/anos-finais/omr", files={"file": ("x.jpg", b"x", "image/jpeg")})
    assert r.status_code == 410
    assert "/anos-finais/omr/objetiva" in r.json()["detail"]
