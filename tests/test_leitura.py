"""
Testes de ponta a ponta do leitor.

A validação não depende de nenhum arquivo gerado: as "fotos" são produzidas em
memória por `make_test_images.py`, com sementes fixas, a partir do PDF oficial
renderizado em `samples/modelo/`. Como o gabarito é conhecido por construção, a
comparação é exata — nada de tolerância a olho.

Cobre:
  - o template continua batendo com o PDF oficial (guarda contra drift);
  - a folha em branco lê tudo BLANK, sem falso positivo;
  - as 6 fotos sintéticas (fácil/médio/difícil, dois fluxos) leem 100%;
  - os três cenários por questão: única, em branco e múltipla;
  - o número do aluno, inclusive com casas em branco à esquerda;
  - foto de cabeça para baixo;
  - página errada no endpoint errado é recusada;
  - a margem entre bolha vazia e bolha marcada continua grande.

Rode com:  .venv/bin/python -m pytest -q
"""
import os
import sys

import cv2
import numpy as np
import pytest

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)

import make_test_images as G                     # noqa: E402
from omr import OMRError, decodificar, ler_fluxo  # noqa: E402
from omr import config as C                       # noqa: E402
from omr import engine as E                       # noqa: E402
from omr import registration as R                 # noqa: E402
from omr import template as T                     # noqa: E402

MODELO = os.path.join(RAIZ, "samples", "modelo")


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def fotos():
    """{nome: (fluxo, bytes_jpeg, gabarito)} — geradas uma vez por sessão."""
    saida = {}
    for nome, fluxo, cenario, semente in G.CASOS:
        gerar = G.gerar_objetiva if fluxo == "objetiva" else G.gerar_redacao
        jpeg, gab = gerar(cenario, semente)
        saida[nome] = (fluxo, jpeg, gab)
    return saida


def _campos_planos(res: dict) -> list[dict]:
    """Todos os campos lidos, seja qual for o fluxo."""
    campos = [q for s in res.get("sections", []) for q in s["results"]]
    campos += list(res.get("correction", {}).values())
    campos += res["student_number"]["digits"]
    return campos


# --------------------------------------------------------------------------- #
# Template x PDF oficial
# --------------------------------------------------------------------------- #
def test_template_bate_com_o_pdf():
    """Se alguém mexer no template (ou reimprimirem a folha), isto acusa."""
    calib = pytest.importorskip("tools.calibrar_template", reason="ferramenta ausente")
    if not os.path.exists(os.path.join(MODELO, "pagina1_300dpi.png")):
        pytest.skip("samples/modelo ausente — rode tools/calibrar_template.py --render")
    medidos = calib.medir_tudo()
    assert calib.conferir(medidos) == 0, "template divergiu do PDF oficial"


def test_geometria_do_template_e_consistente():
    """Nenhuma bolha do template pode cair fora do quadro fiducial."""
    for folha in (T.FOLHA_OBJETIVA, T.FOLHA_REDACAO):
        for bloco in folha.blocos:
            g = bloco.grade
            for li in range(g.n_linhas):
                for ci in range(g.n_colunas):
                    u, v = g.centro(li, ci)
                    assert g.raio < u < 1 - g.raio, f"{bloco.nome}[{li},{ci}] u={u}"
                    assert 0 < v < 1, f"{bloco.nome}[{li},{ci}] v={v}"


# --------------------------------------------------------------------------- #
# Folha limpa (render do PDF)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("pagina, folha", [(1, T.FOLHA_OBJETIVA), (2, T.FOLHA_REDACAO)])
def test_folha_em_branco_nao_inventa_marcacao(pagina, folha):
    caminho = os.path.join(MODELO, f"pagina{pagina}_300dpi.png")
    if not os.path.exists(caminho):
        pytest.skip("samples/modelo ausente")
    reg = R.registrar(cv2.imread(caminho))
    assert reg.origem_fiduciais == "fiduciais"
    leitura = E.ler_folha(reg.canonica, folha)
    assert leitura.cobertura == 1.0, "todas as bolhas impressas deveriam ser achadas"
    for bloco in leitura.blocos.values():
        for campo in bloco.campos:
            assert campo.status == "BLANK", f"{bloco.nome}/{campo.chave} = {campo.status}"
            assert max(campo.fills.values()) < C.PISO_RELATIVO


# --------------------------------------------------------------------------- #
# Fotos sintéticas: leitura exata
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("nome", [c[0] for c in G.CASOS])
def test_foto_sintetica_le_o_gabarito_exato(fotos, nome):
    fluxo, jpeg, gab = fotos[nome]
    res = ler_fluxo(fluxo, decodificar(jpeg))

    assert res["student_number"]["value"] == gab["student_number"], \
        f"número do aluno: raw={res['student_number']['raw']!r}"

    if fluxo == "objetiva":
        for secao in res["sections"]:
            esperado = gab[secao["name"]]
            assert secao["num_questions"] == len(esperado)
            for q in secao["results"]:
                e = esperado[str(q["question"])]
                assert (q["status"], q["answer"]) == (e["status"], e["answer"]), \
                    f"{secao['name']} Q{q['question']} fills={q['fills']}"
    else:
        for chave, e in gab["correction"].items():
            c = res["correction"][chave]
            assert (c["status"], c["value"]) == (e["status"], e["value"]), \
                f"{chave} fills={c['fills']}"


def test_contagem_de_questoes_do_modelo(fotos):
    """A folha tem 25 questões de Linguagens e 26 de Matemática."""
    _, jpeg, _ = fotos["objetiva_facil.jpg"]
    res = ler_fluxo("objetiva", decodificar(jpeg))
    por_area = {s["name"]: s for s in res["sections"]}
    assert por_area["linguagens"]["num_questions"] == 25
    assert por_area["matematica"]["num_questions"] == 26
    assert [q["question"] for q in por_area["linguagens"]["results"]] == list(range(1, 26))
    assert [q["question"] for q in por_area["matematica"]["results"]] == list(range(1, 27))


@pytest.mark.parametrize("status_alvo", ["OK", "BLANK", "MULTIPLE"])
def test_os_tres_cenarios_aparecem(fotos, status_alvo):
    """As fotos de teste exercitam mesmo única/branco/múltipla."""
    _, jpeg, _ = fotos["objetiva_medio.jpg"]
    res = ler_fluxo("objetiva", decodificar(jpeg))
    vistos = [q["status"] for s in res["sections"] for q in s["results"]]
    assert status_alvo in vistos


# --------------------------------------------------------------------------- #
# Número do aluno
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("numero", ["7", "42", "1207", "9876543210"])
@pytest.mark.parametrize("alinhamento", ["direita", "esquerda"])
def test_numero_do_aluno_de_qualquer_tamanho(numero, alinhamento):
    """Número curto vale encostado à direita (convenção) ou à esquerda."""
    rng = np.random.default_rng(99)
    img = G._abrir_modelo(2)
    if alinhamento == "direita":
        G._numero(img, numero, rng)
    else:
        bloco = T.BLOCO_NUMERO_ALUNO
        for i, d in enumerate(numero):
            G.marcar_campo(img, bloco, i, d, rng)
    for bloco in T.BLOCOS_CORRECAO:
        for i, _ in enumerate(bloco.chaves):
            G.marcar_campo(img, bloco, i, bloco.rotulos_de(i)[1], rng)
    jpeg = G.simular_foto(img, rng, **G.CENARIOS["medio"])

    num = ler_fluxo("redacao", decodificar(jpeg))["student_number"]
    assert num["value"] == numero, f"raw={num['raw']!r}"
    assert num["status"] == "OK"
    assert len(num["raw"]) == T.N_DIGITOS
    assert num["raw"].strip("_") == numero


def test_numero_do_aluno_com_buraco_no_meio_nao_e_chutado():
    """Casa em branco no MEIO não pode virar um número plausível."""
    rng = np.random.default_rng(7)
    img = G._abrir_modelo(2)
    bloco = T.BLOCO_NUMERO_ALUNO
    for casa, digito in ((6, "1"), (7, "2"), (9, "4")):     # casa 8 fica vazia
        G.marcar_campo(img, bloco, casa, digito, rng)
    jpeg = G.simular_foto(img, rng, **G.CENARIOS["facil"])

    num = ler_fluxo("redacao", decodificar(jpeg))["student_number"]
    assert num["raw"] == "______12_4"
    assert num["status"] == "AMBIGUOUS"
    assert num["value"] is None


def test_numero_do_aluno_em_branco():
    """Ninguém preencheu o número: status BLANK, e não um número inventado."""
    caminho = os.path.join(MODELO, "pagina2_300dpi.png")
    if not os.path.exists(caminho):
        pytest.skip("samples/modelo ausente")
    res = ler_fluxo("redacao", cv2.imread(caminho))
    assert res["student_number"]["status"] == "BLANK"
    assert res["student_number"]["value"] is None
    assert res["student_number"]["raw"] == "_" * T.N_DIGITOS


# --------------------------------------------------------------------------- #
# Robustez
# --------------------------------------------------------------------------- #
GIROS = {
    0: None,
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


@pytest.mark.parametrize("graus", sorted(GIROS))
def test_foto_em_qualquer_orientacao(fotos, graus):
    """Folha deitada ou de ponta-cabeça lê igual — a orientação é descoberta."""
    _, jpeg, gab = fotos["objetiva_facil.jpg"]
    img = decodificar(jpeg)
    if GIROS[graus] is not None:
        img = cv2.rotate(img, GIROS[graus])

    res = ler_fluxo("objetiva", img)
    # a rotação reportada é a que o motor APLICOU para endireitar a foto
    assert res["alignment"]["rotation"] == (-graus) % 360
    assert res["student_number"]["value"] == gab["student_number"]
    for secao in res["sections"]:
        for q in secao["results"]:
            e = gab[secao["name"]][str(q["question"])]
            assert (q["status"], q["answer"]) == (e["status"], e["answer"]), \
                f"girada {graus}°: {secao['name']} Q{q['question']}"


def test_redacao_deitada(fotos):
    _, jpeg, gab = fotos["redacao_facil.jpg"]
    img = cv2.rotate(decodificar(jpeg), cv2.ROTATE_90_CLOCKWISE)
    res = ler_fluxo("redacao", img)
    assert res["alignment"]["rotation"] == 270
    assert res["student_number"]["value"] == gab["student_number"]
    for chave, e in gab["correction"].items():
        c = res["correction"][chave]
        assert (c["status"], c["value"]) == (e["status"], e["value"])


def test_pagina_errada_no_fluxo_errado(fotos):
    _, jpeg_obj, _ = fotos["objetiva_medio.jpg"]
    _, jpeg_red, _ = fotos["redacao_medio.jpg"]

    with pytest.raises(OMRError, match="/omr/objetiva"):
        ler_fluxo("redacao", decodificar(jpeg_obj))
    with pytest.raises(OMRError, match="/omr/redacao"):
        ler_fluxo("objetiva", decodificar(jpeg_red))


def test_imagem_invalida_da_erro_claro():
    with pytest.raises(OMRError):
        decodificar(b"")
    with pytest.raises(OMRError):
        decodificar(b"isto nao e uma imagem")
    with pytest.raises(OMRError, match="pequena"):
        ler_fluxo("objetiva", np.full((60, 60, 3), 255, np.uint8))


def test_foto_sem_a_folha_e_recusada():
    """Uma parede branca não pode virar um gabarito lido."""
    rng = np.random.default_rng(3)
    ruido = rng.integers(180, 255, (1400, 1000, 3)).astype(np.uint8)
    with pytest.raises(OMRError):
        ler_fluxo("objetiva", ruido)


def test_debug_devolve_imagem_do_mesmo_tamanho(fotos):
    fluxo, jpeg, _ = fotos["redacao_facil.jpg"]
    img = decodificar(jpeg)
    res, anotada = ler_fluxo(fluxo, img, debug=True)
    assert anotada.shape == img.shape
    assert res["flow"] == fluxo
    assert not np.array_equal(anotada, img), "a imagem de debug deveria estar anotada"


# --------------------------------------------------------------------------- #
# Classificação: regra absoluta + regra relativa
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("nome, fills, esperado", [
    # o caso que motivou a regra relativa: marca fraca de verdade, vinda de um
    # PDF real, que antes virava BLANK e perdia a resposta do aluno
    ("marca fraca isolada",
     {"0": 9.5, "1": 6.9, "2": 39.6, "3": 10.1, "4": 13.1,
      "5": 9.5, "6": 12.6, "7": 7.3, "8": 15.0, "9": 13.4}, ("2", "OK", ["2"])),
    ("marca forte", {"A": 100.0, "B": 12.0, "C": 9.0, "D": 14.0}, ("A", "OK", ["A"])),
    # a regra absoluta tem que continuar vindo primeiro, senão a marcação dupla
    # (que a folha manda anular) viraria uma resposta escolhida pela maior
    ("dupla forte", {"A": 100.0, "B": 98.0, "C": 9.0, "D": 14.0},
     (None, "MULTIPLE", ["A", "B"])),
    ("marca forte com rasura ao lado", {"A": 100.0, "B": 45.0, "C": 9.0, "D": 14.0},
     ("A", "REVIEW", ["A"])),
    ("todas vazias", {"A": 11.0, "B": 15.0, "C": 9.0, "D": 14.0}, (None, "BLANK", [])),
    ("a vazia mais escura já medida em folha real",
     {"A": 8.2, "B": 23.3, "C": 13.7, "D": 19.7, "E": 14.7}, (None, "BLANK", [])),
    ("logo abaixo do piso", {"A": 29.0, "B": 12.0, "C": 9.0, "D": 14.0}, (None, "BLANK", [])),
    ("logo acima do piso", {"A": 31.0, "B": 12.0, "C": 9.0, "D": 14.0}, ("A", "OK", ["A"])),
    # sem a exigência de vantagem, isto seria uma escolha no cara-ou-coroa
    ("duas fracas empatadas", {"A": 38.0, "B": 34.0, "C": 9.0, "D": 14.0},
     (None, "REVIEW", [])),
    ("duas fracas com vantagem clara", {"A": 48.0, "B": 30.0, "C": 9.0, "D": 14.0},
     ("A", "OK", ["A"])),
])
def test_classificacao(nome, fills, esperado):
    assert E._classificar(fills) == esperado, nome


def test_piso_por_bloco():
    """O número do aluno tem piso próprio; o resto usa o padrão."""
    assert C.pisos_do_bloco("numero_aluno") == (40.0, 30.0)
    for nome in ("linguagens_b1", "matematica_b2", "correcao_g1", "bloco_inexistente"):
        assert C.pisos_do_bloco(nome) == (C.PISO_RELATIVO, None)


def _digito(maior, segunda=12.0):
    f = {str(d): 5.0 for d in range(10)}
    f["7"], f["3"] = maior, segunda
    return f


@pytest.mark.parametrize("maior, segunda, esperado", [
    (45.0, 12.0, ("7", "OK")),      # passa do piso principal (40)
    (39.6, 12.0, ("7", "OK")),      # cai no fallback (30) — o caso reportado
    (31.0, 12.0, ("7", "OK")),      # ainda no fallback
    (29.0, 12.0, (None, "BLANK")),  # abaixo do fallback
    (45.0, 38.0, (None, "REVIEW")), # passa do piso mas empata: não desce degrau
    (38.0, 34.0, (None, "REVIEW")), # empata dentro do fallback
])
def test_numero_aluno_dois_degraus(maior, segunda, esperado):
    piso, fallback = C.pisos_do_bloco("numero_aluno")
    valor, status, _ = E._classificar(_digito(maior, segunda), piso, fallback)
    assert (valor, status) == esperado


def test_degraus_do_numero_aluno_sao_contiguos():
    """Registro explícito de uma consequência do desenho escolhido.

    Como a exigência de VANTAGEM_RELATIVA vale nos dois degraus e as faixas
    [40,∞) e [30,40) se encostam, o par (40, 30) decide exatamente igual a um
    piso único de 30. O piso de 40 documenta a intenção e fica pronto para ser
    apertado — mas hoje NÃO muda resultado nenhum. Se este teste falhar, é
    porque alguém tornou os degraus realmente diferentes: reveja o README.
    """
    piso, fallback = C.pisos_do_bloco("numero_aluno")
    for maior in np.arange(20.0, 56.0, 0.5):
        for segunda in (5.0, 12.0, maior - 13.0, maior - 11.0):
            f = _digito(float(maior), float(max(0.0, segunda)))
            assert E._classificar(f, piso, fallback) == E._classificar(f, fallback, None)


def test_marcadas_sempre_batem_com_o_valor():
    """`marked` e `value` não podem se contradizer — o cliente usa os dois."""
    casos = [
        {"A": 100.0, "B": 9.0, "C": 8.0, "D": 7.0},
        {"A": 39.6, "B": 9.0, "C": 8.0, "D": 7.0},
        {"A": 100.0, "B": 97.0, "C": 8.0, "D": 7.0},
        {"A": 11.0, "B": 9.0, "C": 8.0, "D": 7.0},
        {"A": 38.0, "B": 34.0, "C": 8.0, "D": 7.0},
    ]
    for fills in casos:
        valor, status, marcadas = E._classificar(fills)
        if status == "OK":
            assert marcadas == [valor]
        elif status == "BLANK":
            assert marcadas == [] and valor is None
        elif status == "MULTIPLE":
            assert len(marcadas) >= 2 and valor is None
        # em REVIEW o valor pode vir preenchido (marca clara + rasura) ou não
        if valor is not None:
            assert valor in marcadas


# --------------------------------------------------------------------------- #
# Margem de decisão
# --------------------------------------------------------------------------- #
def test_margem_entre_bolha_vazia_e_marcada(fotos):
    """O que separa acerto de chute é esta margem — se ela encolher, o limiar
    em `config.MARK_THRESHOLD` precisa ser revisto (e não o teste)."""
    vazias, marcadas = [], []
    for nome, (fluxo, jpeg, _) in fotos.items():
        res = ler_fluxo(fluxo, decodificar(jpeg))
        for campo in _campos_planos(res):
            for rotulo, valor in campo["fills"].items():
                (marcadas if rotulo in campo["marked"] else vazias).append(valor)

    vazias, marcadas = np.array(vazias), np.array(marcadas)
    assert vazias.max() < C.PISO_RELATIVO, f"bolha vazia chegou a {vazias.max():.1f}%"
    assert marcadas.min() > C.MARK_THRESHOLD + 15, f"marcada mais fraca: {marcadas.min():.1f}%"
    assert marcadas.min() - vazias.max() > 40, "margem de decisão encolheu"


def test_alinhamento_reportado(fotos):
    """O campo `alignment` precisa dizer a verdade sobre o registro."""
    for nome, (fluxo, jpeg, _) in fotos.items():
        res = ler_fluxo(fluxo, decodificar(jpeg))
        a = res["alignment"]
        assert a["fiducials"] == "fiduciais", f"{nome}: {a['fiducials']}"
        assert a["coverage"] > 0.85, f"{nome}: cobertura {a['coverage']}"
        assert a["global_fit"] == "afim"
        for nome_bloco, b in a["blocks"].items():
            assert b["matched"] <= b["expected"]
            assert b["mode"].startswith("afim"), f"{nome}/{nome_bloco}: {b['mode']}"
