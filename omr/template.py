"""
Modelo declarativo da folha — Cartão-Resposta Veloz, Anos Finais.

Todas as coordenadas são NORMALIZADAS no **quadro fiducial**: o retângulo cujos
cantos são os centros dos 4 marcadores fiduciais impressos na folha.

    (0, 0) = centro do fiducial superior-esquerdo
    (1, 1) = centro do fiducial inferior-direito

Com isso o template independe de DPI, do tamanho da foto e de quanta margem de
papel o aluno deixou aparecer — só depende do desenho da folha.

Os números abaixo foram MEDIDOS automaticamente no PDF oficial renderizado a
300 dpi (rode `tools/calibrar_template.py` para conferir/regerar). O resíduo do
ajuste linear ficou < 0,1 px em todas as grades: a folha é perfeitamente
regular, então uma grade (origem + passo) descreve cada bloco exatamente.

Dois MODELOS publicados, com a mesma geometria de base:

  - MODELO_ANOS_INICIAIS — objetiva: Linguagens 1..21, Matemática 1..22.
    Não tem página de produção de texto.
  - MODELO_ANOS_FINAIS   — objetiva: Linguagens 1..25, Matemática 1..26;
    redação: quadro de correção do professor.

Os dois compartilham cabeçalho (nº do aluno), passo entre bolhas, raio e a
posição horizontal dos blocos. Muda só quantas questões cada disciplina tem e,
por causa disso, a altura em que os blocos de questão começam.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# --------------------------------------------------------------------------- #
# Geometria do quadro fiducial
# --------------------------------------------------------------------------- #
# Medido no render 300 dpi: fiduciais em (82,0; 82,05) e (2468,0; 3495,55).
FIDUCIAL_W_PX = 2386.0        # distância horizontal entre centros dos fiduciais
FIDUCIAL_H_PX = 3413.5        # distância vertical
ASPECTO = FIDUCIAL_H_PX / FIDUCIAL_W_PX      # ~1,43064 (altura / largura)

# Tamanho da marca fiducial (lado do "+" externo) em fração da largura do quadro.
FIDUCIAL_MARCA_FRAC = 89.0 / FIDUCIAL_W_PX   # ~0,0373

# Posição do centro de cada fiducial em fração da PÁGINA inteira (papel).
# Usado só para saber onde procurar as marcas antes de conhecer o quadro.
# Página A4-ish do PDF: 2552 x 3579 px a 300 dpi.
FIDUCIAL_NA_PAGINA = (82.0 / 2552.0, 82.05 / 3579.0)   # (u, v) do canto TL
PAGINA_ASPECTO = 3579.0 / 2552.0                       # altura / largura do papel


# --------------------------------------------------------------------------- #
# Blocos
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Grade:
    """Grade regular de bolhas, em coordenadas normalizadas do quadro fiducial.

    `(u0, v0)` é o centro da bolha [linha 0, coluna 0]; `du`/`dv` são os passos
    entre colunas e entre linhas. `raio` é o raio da bolha impressa, também em
    fração da LARGURA do quadro (para não distorcer com o aspecto).
    """

    u0: float
    v0: float
    du: float
    dv: float
    n_linhas: int
    n_colunas: int
    raio: float

    def centro(self, linha: int, coluna: int) -> tuple[float, float]:
        return (self.u0 + coluna * self.du, self.v0 + linha * self.dv)

    def centros(self) -> list[list[tuple[float, float]]]:
        return [
            [self.centro(li, ci) for ci in range(self.n_colunas)]
            for li in range(self.n_linhas)
        ]


@dataclass(frozen=True)
class Bloco:
    """Conjunto de campos que compartilham uma grade (e um ROI de alinhamento).

    - `eixo="linha"`: cada LINHA da grade é um campo (ex.: uma questão A–D);
    - `eixo="coluna"`: cada COLUNA da grade é um campo (ex.: um dígito do número
      do aluno, cujas 10 opções ficam empilhadas na vertical).

    `chaves` nomeia os campos na ordem do eixo; `rotulos` são os rótulos das
    opções de cada campo (idem ordem). `rotulos_por_campo` permite que campos do
    mesmo bloco tenham rótulos diferentes (o quadro de correção tem uma linha
    A–E e outra 0–4).
    """

    nome: str
    grade: Grade
    eixo: str                        # "linha" | "coluna"
    chaves: tuple[str, ...]
    rotulos: tuple[str, ...]
    rotulos_por_campo: tuple[tuple[str, ...], ...] | None = None

    def __post_init__(self) -> None:
        n = self.grade.n_linhas if self.eixo == "linha" else self.grade.n_colunas
        n_op = self.grade.n_colunas if self.eixo == "linha" else self.grade.n_linhas
        if len(self.chaves) != n:
            raise ValueError(f"{self.nome}: {len(self.chaves)} chaves para {n} campos")
        if len(self.rotulos) != n_op:
            raise ValueError(f"{self.nome}: {len(self.rotulos)} rótulos para {n_op} opções")
        if self.rotulos_por_campo is not None and len(self.rotulos_por_campo) != n:
            raise ValueError(f"{self.nome}: rotulos_por_campo com tamanho errado")

    def rotulos_de(self, indice_campo: int) -> tuple[str, ...]:
        if self.rotulos_por_campo is None:
            return self.rotulos
        return self.rotulos_por_campo[indice_campo]

    def celulas_do_campo(self, indice_campo: int) -> list[tuple[int, int]]:
        """(linha, coluna) de cada opção do campo, na ordem dos rótulos."""
        if self.eixo == "linha":
            return [(indice_campo, c) for c in range(self.grade.n_colunas)]
        return [(l, indice_campo) for l in range(self.grade.n_linhas)]


@dataclass(frozen=True)
class Folha:
    """Uma página de um modelo: os blocos que devem ser lidos nela.

    `areas` só existe na página objetiva e diz quais blocos compõem cada
    disciplina, na ordem das questões (Linguagens = bloco 1 + bloco 2, etc.).
    """

    nome: str                                  # "objetiva" | "redacao"
    titulo: str
    blocos: tuple[Bloco, ...]
    areas: dict[str, tuple[Bloco, ...]] = field(default_factory=dict)

    @property
    def blocos_de_questao(self) -> tuple[Bloco, ...]:
        return tuple(b for blocos in self.areas.values() for b in blocos)


@dataclass(frozen=True)
class Modelo:
    """Um modelo de folha da coleção (Anos Iniciais, Anos Finais, ...).

    Os dois modelos publicados são a MESMA folha com contagens diferentes: o
    cabeçalho, o passo entre bolhas, o raio e a posição horizontal dos blocos
    são idênticos; muda só quantas questões cada disciplina tem e, por causa
    disso, a altura em que os blocos começam. Anos Iniciais não tem a página de
    produção de texto — por isso `folhas` é um dicionário e não um par fixo.
    """

    nome: str
    titulo: str
    folhas: dict[str, Folha]

    def folha(self, fluxo: str) -> Folha | None:
        return self.folhas.get(fluxo)


# --------------------------------------------------------------------------- #
# Número do aluno — idêntico nas duas páginas
# --------------------------------------------------------------------------- #
N_DIGITOS = 10
DIGITOS = tuple(str(d) for d in range(10))

GRADE_NUMERO_ALUNO = Grade(
    u0=0.624396, v0=0.129596,
    du=0.038109, dv=0.020089,
    n_linhas=10, n_colunas=N_DIGITOS,     # 10 linhas = dígitos 0..9
    raio=0.011220,
)

BLOCO_NUMERO_ALUNO = Bloco(
    nome="numero_aluno",
    grade=GRADE_NUMERO_ALUNO,
    eixo="coluna",                        # cada coluna é uma casa do número
    chaves=tuple(f"casa_{i + 1}" for i in range(N_DIGITOS)),
    rotulos=DIGITOS,
)


# --------------------------------------------------------------------------- #
# Página objetiva — CARTÃO-RESPOSTA
# --------------------------------------------------------------------------- #
ALTERNATIVAS = ("A", "B", "C", "D")

# Geometria COMPARTILHADA pelos dois modelos. Medida no PDF oficial dos Anos
# Finais e reconferida no scan dos Anos Iniciais: as quatro colunas de blocos
# caem no mesmo lugar (diferença < 1,4 px em 2386) e o passo é o mesmo.
_RESP_DV = 0.024388                # passo entre linhas (questões)
_RESP_DU = 0.049740                # passo entre alternativas
_RESP_RAIO = 0.014823
_RESP_U0 = (0.068376, 0.306551, 0.567638, 0.805821)   # 4 blocos, esq -> dir

# O que MUDA de um modelo para o outro: quantas questões cada disciplina tem e,
# como consequência do layout, a altura da primeira linha.
_V0_ANOS_FINAIS = 0.680303         # medido no PDF oficial (resíduo < 0,1 px)
_V0_ANOS_INICIAIS = 0.708603       # medido no scan de referência (resíduo ~1 px)


def _bloco_questoes(nome: str, u0: float, v0: float, primeira: int, n: int) -> Bloco:
    return Bloco(
        nome=nome,
        grade=Grade(
            u0=u0, v0=v0, du=_RESP_DU, dv=_RESP_DV,
            n_linhas=n, n_colunas=len(ALTERNATIVAS), raio=_RESP_RAIO,
        ),
        eixo="linha",
        chaves=tuple(str(primeira + i) for i in range(n)),
        rotulos=ALTERNATIVAS,
    )


def _folha_objetiva(titulo: str, v0: float, linguagens: tuple[int, int],
                    matematica: tuple[int, int]) -> Folha:
    """Monta a página objetiva de um modelo.

    `linguagens`/`matematica` são (nº de questões no BLOCO 1, no BLOCO 2). O
    bloco 2 continua a numeração de onde o bloco 1 parou.
    """
    areas: dict[str, tuple[Bloco, ...]] = {}
    for i, (area, (n1, n2)) in enumerate(
        (("linguagens", linguagens), ("matematica", matematica))
    ):
        u_b1, u_b2 = _RESP_U0[2 * i], _RESP_U0[2 * i + 1]
        areas[area] = (
            _bloco_questoes(f"{area}_b1", u_b1, v0, 1, n1),
            _bloco_questoes(f"{area}_b2", u_b2, v0, n1 + 1, n2),
        )
    blocos = (BLOCO_NUMERO_ALUNO,) + tuple(b for bs in areas.values() for b in bs)
    return Folha(nome="objetiva", titulo=titulo, blocos=blocos, areas=areas)


# --------------------------------------------------------------------------- #
# Página 2 — PRODUÇÃO DE TEXTO (quadro de correção do professor)
# --------------------------------------------------------------------------- #
SITUACOES = ("A", "B", "C", "D", "E")
NIVEIS = ("0", "1", "2", "3", "4")

_CORR_V = (0.960587, 0.980624)          # duas linhas do quadro
_CORR_DV = _CORR_V[1] - _CORR_V[0]
_CORR_DU = 0.031213
_CORR_RAIO = 0.011220
_CORR_U0 = (0.202294, 0.502470, 0.802644)   # três grupos de 5 bolhas

# Layout impresso: cada grupo tem a linha de cima e a de baixo.
#   grupo 1 -> Situação (A–E)   / Competência 01 (0–4)
#   grupo 2 -> Competência 02   / Competência 03
#   grupo 3 -> Competência 04   / Competência 05
_CORR_CAMPOS = (
    (("situacao", SITUACOES), ("competencia_01", NIVEIS)),
    (("competencia_02", NIVEIS), ("competencia_03", NIVEIS)),
    (("competencia_04", NIVEIS), ("competencia_05", NIVEIS)),
)


def _bloco_correcao(indice: int) -> Bloco:
    campos = _CORR_CAMPOS[indice]
    return Bloco(
        nome=f"correcao_g{indice + 1}",
        grade=Grade(
            u0=_CORR_U0[indice], v0=_CORR_V[0], du=_CORR_DU, dv=_CORR_DV,
            n_linhas=2, n_colunas=5, raio=_CORR_RAIO,
        ),
        eixo="linha",
        chaves=tuple(c[0] for c in campos),
        rotulos=NIVEIS,                       # tamanho de referência (5 opções)
        rotulos_por_campo=tuple(c[1] for c in campos),
    )


CORRECAO_G1 = _bloco_correcao(0)
CORRECAO_G2 = _bloco_correcao(1)
CORRECAO_G3 = _bloco_correcao(2)

BLOCOS_CORRECAO = (CORRECAO_G1, CORRECAO_G2, CORRECAO_G3)

# Ordem em que os campos de correção aparecem na resposta da API.
ORDEM_CORRECAO = (
    "situacao",
    "competencia_01",
    "competencia_02",
    "competencia_03",
    "competencia_04",
    "competencia_05",
)

FOLHA_REDACAO = Folha(
    nome="redacao",
    titulo="PRODUÇÃO DE TEXTO — Anos finais",
    blocos=(BLOCO_NUMERO_ALUNO, CORRECAO_G1, CORRECAO_G2, CORRECAO_G3),
)


# --------------------------------------------------------------------------- #
# Os modelos publicados
# --------------------------------------------------------------------------- #
MODELO_ANOS_FINAIS = Modelo(
    nome="anos_finais",
    titulo="Cartão-Resposta Veloz — Anos Finais",
    folhas={
        "objetiva": _folha_objetiva(
            "CARTÃO-RESPOSTA — Anos finais", _V0_ANOS_FINAIS,
            linguagens=(13, 12),      # 1..13 + 14..25 = 25 questões
            matematica=(13, 13),      # 1..13 + 14..26 = 26 questões
        ),
        "redacao": FOLHA_REDACAO,
    },
)

# Anos Iniciais não tem página de produção de texto — só a objetiva.
MODELO_ANOS_INICIAIS = Modelo(
    nome="anos_iniciais",
    titulo="Cartão-Resposta Veloz — Anos Iniciais",
    folhas={
        "objetiva": _folha_objetiva(
            "CARTÃO-RESPOSTA — Anos iniciais", _V0_ANOS_INICIAIS,
            linguagens=(11, 10),      # 1..11 + 12..21 = 21 questões
            matematica=(11, 11),      # 1..11 + 12..22 = 22 questões
        ),
    },
)

MODELOS = {m.nome: m for m in (MODELO_ANOS_INICIAIS, MODELO_ANOS_FINAIS)}

#: Quais modelos atendem cada fluxo.
MODELOS_POR_FLUXO = {
    fluxo: tuple(m for m in MODELOS.values() if fluxo in m.folhas)
    for fluxo in ("objetiva", "redacao")
}

#: Todo bloco de todos os modelos, por nome — usado para desenhar o debug.
TODOS_OS_BLOCOS = {
    b.nome: b
    for m in MODELOS.values()
    for f in m.folhas.values()
    for b in f.blocos
}

# Compatibilidade: o modelo dos Anos Finais como estava antes do registry.
FOLHA_OBJETIVA = MODELO_ANOS_FINAIS.folhas["objetiva"]
AREAS = FOLHA_OBJETIVA.areas
FOLHAS = {FOLHA_OBJETIVA.nome: FOLHA_OBJETIVA, FOLHA_REDACAO.nome: FOLHA_REDACAO}
