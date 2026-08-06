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
from omr import OMRError, decodificar, ler_arquivo, ler_fluxo  # noqa: E402
from omr import config as C                       # noqa: E402
from omr import engine as E                       # noqa: E402
from omr import registration as R                 # noqa: E402
from omr import template as T                     # noqa: E402

MODELO = os.path.join(RAIZ, "samples", "modelo")


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
#: casos cujo render de referência já existe em samples/modelo/
CASOS_DISPONIVEIS = [c for c in G.CASOS if G.caso_disponivel(c)]


@pytest.fixture(scope="module")
def fotos():
    """{nome: (fluxo, bytes_jpeg, gabarito)} — geradas uma vez por sessão."""
    saida = {}
    for nome, fluxo, cenario, semente, modelo in CASOS_DISPONIVEIS:
        gerar = G.gerar_objetiva if fluxo == "objetiva" else G.gerar_redacao
        jpeg, gab = gerar(cenario, semente, modelo)
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
    for modelo in T.MODELOS.values():
      for folha in modelo.folhas.values():
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
@pytest.mark.parametrize("pagina, folha", [
    (1, T.MODELO_ANOS_FINAIS.folhas["objetiva"]),
    (2, T.MODELO_ANOS_FINAIS.folhas["redacao"]),
])
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
            assert max(campo.fills.values()) < C.REVIEW_LOW


# --------------------------------------------------------------------------- #
# Fotos sintéticas: leitura exata
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("nome", [c[0] for c in CASOS_DISPONIVEIS])
def test_foto_sintetica_le_o_gabarito_exato(fotos, nome):
    fluxo, jpeg, gab = fotos[nome]
    res = ler_fluxo(fluxo, decodificar(jpeg))

    assert res["model"] == gab["modelo"], f"modelo detectado errado em {nome}"
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
    _, jpeg, _ = fotos["finais_objetiva_facil.jpg"]
    res = ler_fluxo("objetiva", decodificar(jpeg))
    por_area = {s["name"]: s for s in res["sections"]}
    assert por_area["linguagens"]["num_questions"] == 25
    assert por_area["matematica"]["num_questions"] == 26
    assert [q["question"] for q in por_area["linguagens"]["results"]] == list(range(1, 26))
    assert [q["question"] for q in por_area["matematica"]["results"]] == list(range(1, 27))


@pytest.mark.parametrize("status_alvo", ["OK", "BLANK", "MULTIPLE"])
def test_os_tres_cenarios_aparecem(fotos, status_alvo):
    """As fotos de teste exercitam mesmo única/branco/múltipla."""
    _, jpeg, _ = fotos["finais_objetiva_medio.jpg"]
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
    img = G._abrir_modelo("anos_finais", "redacao")
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
    img = G._abrir_modelo("anos_finais", "redacao")
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
    _, jpeg, gab = fotos["finais_objetiva_facil.jpg"]
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
    _, jpeg, gab = fotos["finais_redacao_facil.jpg"]
    img = cv2.rotate(decodificar(jpeg), cv2.ROTATE_90_CLOCKWISE)
    res = ler_fluxo("redacao", img)
    assert res["alignment"]["rotation"] == 270
    assert res["student_number"]["value"] == gab["student_number"]
    for chave, e in gab["correction"].items():
        c = res["correction"][chave]
        assert (c["status"], c["value"]) == (e["status"], e["value"])


def test_pagina_errada_no_fluxo_errado(fotos):
    _, jpeg_obj, _ = fotos["finais_objetiva_medio.jpg"]
    _, jpeg_red, _ = fotos["finais_redacao_medio.jpg"]

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
    fluxo, jpeg, _ = fotos["finais_redacao_facil.jpg"]
    img = decodificar(jpeg)
    res, anotada = ler_fluxo(fluxo, img, debug=True)
    assert anotada.shape == img.shape
    assert res["flow"] == fluxo
    assert not np.array_equal(anotada, img), "a imagem de debug deveria estar anotada"


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
    assert vazias.max() < C.REVIEW_LOW, f"bolha vazia chegou a {vazias.max():.1f}%"
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


# --------------------------------------------------------------------------- #
# Modelos de folha (Anos Iniciais x Anos Finais)
# --------------------------------------------------------------------------- #
FOLHAS_REAIS = {
    "anos_iniciais": ("/Users/lucas/Desktop/Screenshot 2026-08-03 at 09.09.15.png", "objetiva"),
    "anos_finais": ("/Users/lucas/Desktop/Screenshot 2026-07-30 at 15.29.41.png", "objetiva"),
}


def test_registry_de_modelos():
    """O que cada modelo declara tem de bater com a folha impressa."""
    ini = T.MODELOS["anos_iniciais"]
    fin = T.MODELOS["anos_finais"]

    assert set(ini.folhas) == {"objetiva"}, "Anos Iniciais não tem produção de texto"
    assert set(fin.folhas) == {"objetiva", "redacao"}

    contagens = {
        "anos_iniciais": {"linguagens": 21, "matematica": 22},
        "anos_finais": {"linguagens": 25, "matematica": 26},
    }
    for nome, esperado in contagens.items():
        folha = T.MODELOS[nome].folhas["objetiva"]
        for area, n in esperado.items():
            chaves = [c for b in folha.areas[area] for c in b.chaves]
            assert len(chaves) == n, f"{nome}/{area}"
            assert chaves == [str(i) for i in range(1, n + 1)], f"{nome}/{area} numeração"


def test_modelos_compartilham_a_geometria_de_base():
    """Só a altura das linhas e a contagem mudam — o resto é a mesma folha.

    Se este teste falhar, ou a Editora mudou o layout, ou alguém calibrou um
    modelo com números de outro. Nos dois casos, medir de novo é obrigatório.
    """
    ini = T.MODELOS["anos_iniciais"].folhas["objetiva"]
    fin = T.MODELOS["anos_finais"].folhas["objetiva"]

    for b_ini, b_fin in zip(ini.blocos_de_questao, fin.blocos_de_questao):
        gi, gf = b_ini.grade, b_fin.grade
        assert gi.u0 == gf.u0, f"{b_ini.nome}: coluna do bloco mudou"
        assert gi.du == gf.du and gi.dv == gf.dv, f"{b_ini.nome}: passo mudou"
        assert gi.raio == gf.raio, f"{b_ini.nome}: raio mudou"
    assert ini.blocos_de_questao[0].grade.v0 != fin.blocos_de_questao[0].grade.v0

    # o cabeçalho é literalmente o mesmo objeto nos dois modelos
    assert T.BLOCO_NUMERO_ALUNO in ini.blocos and T.BLOCO_NUMERO_ALUNO in fin.blocos


@pytest.mark.parametrize("modelo", sorted(FOLHAS_REAIS))
def test_modelo_reconhecido_em_folha_real(modelo):
    """A troca de modelo é a falha mais perigosa do sistema: os dois têm o mesmo
    passo entre linhas, então o template errado NÃO quebra — ele devolve
    respostas deslocadas uma questão. Por isso a margem é conferida aqui."""
    caminho, fluxo = FOLHAS_REAIS[modelo]
    if not os.path.exists(caminho):
        pytest.skip(f"{caminho} ausente")

    reg = R.registrar(ler_arquivo(caminho))
    achado, diag = E.identificar_modelo(reg.canonica, fluxo)
    assert achado is not None and achado.nome == modelo, diag

    custos = sorted(diag["custos"].values())
    assert custos[0] <= 0.5, f"encaixe folgado demais: {diag}"
    assert custos[1] - custos[0] >= 2.0, f"margem estreita entre modelos: {diag}"


def test_folha_de_iniciais_le_certo_de_ponta_a_ponta():
    """Regressão da folha real: 43 questões, todas resolvidas, nº do aluno inteiro.

    Ela é o caso difícil do acervo — o aluno CIRCULOU as bolhas em vez de
    pintar, então os preenchimentos caem em 39%..76% e quem decide é a regra
    relativa, não o limiar absoluto.
    """
    caminho, _ = FOLHAS_REAIS["anos_iniciais"]
    if not os.path.exists(caminho):
        pytest.skip(f"{caminho} ausente")

    res = ler_fluxo("objetiva", ler_arquivo(caminho))
    assert res["model"] == "anos_iniciais"
    assert res["student_number"]["value"] == "2026005882"
    assert res["summary"] == {"ok": 43, "blank": 0, "multiple": 0, "review": 0}

    esperado = {"linguagens": "BADBCBAACDBCABDABDBAB",
                "matematica": "BACDABCDABBACBABBBDABC"}
    for sec in res["sections"]:
        lido = "".join(q["answer"] or "-" for q in sec["results"])
        assert lido == esperado[sec["name"]], sec["name"]


def test_marca_circulada_depende_da_regra_relativa():
    """Sem a regra relativa a folha circulada perde respostas — este teste fixa
    o porquê de ela existir, com os números medidos na folha real."""
    piso_q = C.pisos_do_bloco("linguagens_b1")
    piso_n = C.pisos_do_bloco("numero_aluno")

    # medidos na folha real de Anos Iniciais: vencedora isolada, abaixo de 55
    assert E._classificar({"A": 7.2, "B": 54.0, "C": 5.9, "D": 10.5}, *piso_q)[:2] == ("B", "OK")
    assert E._classificar({"A": 52.6, "B": 10.2, "C": 7.1, "D": 9.0}, *piso_q)[:2] == ("A", "OK")
    assert E._classificar({"2": 39.3, "8": 15.0, "9": 13.4, "0": 9.0}, *piso_n)[:2] == ("2", "OK")

    # e a marca abandonada ao lado da resposta continua sendo ignorada
    assert E._classificar({"A": 8.0, "B": 26.1, "C": 9.0, "D": 64.7}, *piso_q)[:2] == ("D", "OK")
