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

Duas páginas, dois fluxos:
  - FOLHA_OBJETIVA (página 1) — Linguagens 1..25, Matemática 1..26, nº do aluno;
  - FOLHA_REDACAO  (página 2) — quadro de correção do professor, nº do aluno.
"""
from __future__ import annotations

from dataclasses import dataclass

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
    """Uma página do modelo: os blocos que devem ser lidos nela."""

    nome: str
    titulo: str
    blocos: tuple[Bloco, ...]


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
# Página 1 — CARTÃO-RESPOSTA (Linguagens 1..25, Matemática 1..26)
# --------------------------------------------------------------------------- #
ALTERNATIVAS = ("A", "B", "C", "D")

# Todos os 4 blocos de questões começam na mesma altura e têm o mesmo passo.
_RESP_V0 = 0.680303
_RESP_DV = 0.024388
_RESP_DU = 0.049740
_RESP_RAIO = 0.014823


def _bloco_questoes(nome: str, u0: float, primeira: int, n: int) -> Bloco:
    return Bloco(
        nome=nome,
        grade=Grade(
            u0=u0, v0=_RESP_V0, du=_RESP_DU, dv=_RESP_DV,
            n_linhas=n, n_colunas=len(ALTERNATIVAS), raio=_RESP_RAIO,
        ),
        eixo="linha",
        chaves=tuple(str(primeira + i) for i in range(n)),
        rotulos=ALTERNATIVAS,
    )


# Bloco 1 = questões 1..13; Bloco 2 = 14..25 (Linguagens) / 14..26 (Matemática).
LINGUAGENS_B1 = _bloco_questoes("linguagens_b1", 0.068376, 1, 13)
LINGUAGENS_B2 = _bloco_questoes("linguagens_b2", 0.306551, 14, 12)
MATEMATICA_B1 = _bloco_questoes("matematica_b1", 0.567638, 1, 13)
MATEMATICA_B2 = _bloco_questoes("matematica_b2", 0.805821, 14, 13)

# Quais blocos compõem cada área (na ordem das questões).
AREAS = {
    "linguagens": (LINGUAGENS_B1, LINGUAGENS_B2),
    "matematica": (MATEMATICA_B1, MATEMATICA_B2),
}

FOLHA_OBJETIVA = Folha(
    nome="objetiva",
    titulo="CARTÃO-RESPOSTA — Anos finais",
    blocos=(BLOCO_NUMERO_ALUNO, LINGUAGENS_B1, LINGUAGENS_B2, MATEMATICA_B1, MATEMATICA_B2),
)


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


FOLHAS = {FOLHA_OBJETIVA.nome: FOLHA_OBJETIVA, FOLHA_REDACAO.nome: FOLHA_REDACAO}
