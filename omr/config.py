"""
Parâmetros de detecção e limiares de decisão do leitor.

A GEOMETRIA da folha não mora aqui — ela está em `omr/template.py`, em
coordenadas normalizadas pelo quadro dos 4 marcadores fiduciais. Aqui ficam só
os parâmetros do processamento de imagem e os cortes de decisão.

Pipeline (ver `omr/registration.py` e `omr/engine.py`):
    foto -> acha o papel -> acha os 4 fiduciais -> homografia -> folha canônica
         -> por bloco: reajusta a grade nas bolhas reais -> mede -> classifica
"""

# --------------------------------------------------------------------------- #
# Espaço canônico (a folha já retificada)
# --------------------------------------------------------------------------- #
# Largura, em pixels, do QUADRO FIDUCIAL na imagem canônica. Define a resolução
# de trabalho: com 1700 px a bolha de resposta fica com raio ~25 px, que é a
# escala em que os limiares abaixo foram calibrados.
CANON_W = 1700
# Margem extra além do quadro fiducial, em fração da largura do quadro. Serve
# para não cortar bolha nenhuma quando o registro sai alguns pixels torto.
CANON_MARGEM = 0.025

# --------------------------------------------------------------------------- #
# Localização do papel (etapa grosseira, antes dos fiduciais)
# --------------------------------------------------------------------------- #
PAPEL_LARGURA_TRABALHO = 1000     # downscale usado só para achar o papel
PAPEL_AREA_MIN_FRAC = 0.20        # o papel precisa ocupar ao menos isso da foto
PAGINA_LARGURA = 1500             # largura da página retificada grosseira

# --------------------------------------------------------------------------- #
# Detecção dos marcadores fiduciais
# --------------------------------------------------------------------------- #
# Janela de busca ao redor da posição esperada de cada marca, em fração da
# largura da página. Generosa o bastante para absorver erro do passo anterior.
FIDUCIAL_JANELA = 0.11
# Escalas testadas no template matching, relativas ao tamanho esperado da marca.
FIDUCIAL_ESCALAS = (0.72, 0.85, 1.0, 1.18, 1.4)
# Correlação normalizada mínima para aceitar uma marca.
FIDUCIAL_SCORE_MIN = 0.42
# Tolerância do formato do quadrilátero formado pelas 4 marcas: o aspecto medido
# não pode fugir mais do que isso (em fração) do aspecto nominal da folha.
FIDUCIAL_ASPECTO_TOL = 0.18

# --------------------------------------------------------------------------- #
# Normalização de iluminação (flat-field)
# --------------------------------------------------------------------------- #
# Divide o recorte por uma versão muito borrada dele: mata sombra, gradiente de
# luz e o fundo colorido das tarjas impressas. Sigma em px do espaço canônico.
FLATFIELD_SIGMA = 41

# --------------------------------------------------------------------------- #
# Detecção das bolhas para reajuste fino da grade
# --------------------------------------------------------------------------- #
ADAPT_BLOCK = 41                  # blockSize do adaptiveThreshold
ADAPT_C = 8                       # constante C do adaptiveThreshold
RAIO_MIN_FRAC = 0.62              # faixa de raio aceita, relativa ao raio do
RAIO_MAX_FRAC = 1.55              #   template
CIRCULARIDADE_MIN = 0.62          # 4*pi*A/P^2 — descarta quadrados e texto
PREENCH_CIRCULO_MIN = 0.55        # área / área do círculo circunscrito

# Reajuste da grade (snap) sobre as bolhas realmente detectadas:
SNAP_TOL_FRAC = 0.45              # casa candidata com posição esperada se a
                                  #   distância for < isso * menor passo
SNAP_MIN_PARES_PAGINA = 24        # pares mínimos para o ajuste afim global
SNAP_MIN_PARES_BLOCO = 6          # pares mínimos para o ajuste local do bloco
SNAP_ESCALA_TOL = 0.12            # afim global só vale se a escala ficar em 1±isso
SNAP_DESLOC_MAX_FRAC = 0.50       # deslocamento local no máximo isso * passo

# --------------------------------------------------------------------------- #
# Medição do preenchimento
# --------------------------------------------------------------------------- #
# Amostra um disco menor que a bolha impressa, para não contar o anel do
# círculo nem o dígito impresso dentro dele como se fossem marca.
AMOSTRA_R_FRAC = 0.72
# Nível de tinta no recorte JÁ normalizado por flat-field (papel ~= 255).
# Pixel abaixo disso conta como escuro. Limiar fixo — e não Otsu — porque um
# bloco totalmente em branco (comum no quadro de correção) faz o Otsu inventar
# uma separação onde só existe ruído de papel.
NIVEL_TINTA = 160

# --------------------------------------------------------------------------- #
# Limiares de decisão (percentual de pixels escuros dentro da bolha)
# --------------------------------------------------------------------------- #
# Separação medida na folha modelo, nas fotos sintéticas e em digitalizações
# reais: bolha VAZIA fica em 0%..24%; bolha bem pintada, em 80%..100%.
#
# --- regra ABSOLUTA: decide sozinha e é a única que enxerga marcação dupla ---
MARK_THRESHOLD = 55.0             # fill >= isto  => bolha MARCADA
REVIEW_LOW = 40.0                 # bolha nesta faixa ao lado de uma marcada
                                  #   (rasura, tentativa de apagar) => REVIEW

# --- regra RELATIVA: só entra quando NENHUMA bolha cruzou o limiar acima ---
# Marca fraca (caneta leve, traço pequeno, marca grande que o flat-field
# clareia) não pode virar folha em branco. Aí a mais preenchida vence, com dois
# guarda-corpos:
PISO_RELATIVO = 30.0              # abaixo disto para TODAS => BLANK de verdade.
                                  #   A bolha vazia mais escura já medida foi
                                  #   23,3% — a folga aqui é de ~7 pontos.
VANTAGEM_RELATIVA = 12.0          # a vencedora precisa estar isso à frente da
                                  #   segunda; senão é empate técnico => REVIEW.
                                  #   Entre bolhas todas vazias a diferença
                                  #   típica é < 10 pontos.

# Piso próprio de alguns blocos. O número do aluno é o caso: um dígito errado
# troca a identidade da prova inteira, e a coluna tem 10 opções (mais chance de
# vizinha suja) contra as 4 de uma questão. Daí um degrau "confiante" mais alto,
# com um segundo degrau abaixo dele para não perder marca fraca.
PISO_RELATIVO_POR_BLOCO = {"numero_aluno": 40.0}
PISO_FALLBACK_POR_BLOCO = {"numero_aluno": 30.0}


def pisos_do_bloco(nome: str) -> tuple[float, float | None]:
    """(piso principal, piso de fallback) do bloco. Fallback None = sem 2º degrau."""
    return (
        PISO_RELATIVO_POR_BLOCO.get(nome, PISO_RELATIVO),
        PISO_FALLBACK_POR_BLOCO.get(nome),
    )

# --------------------------------------------------------------------------- #
# Identificação da página (evita ler a folha errada no endpoint errado)
# --------------------------------------------------------------------------- #
# A página objetiva tem 204 bolhas grandes na metade de baixo; a de redação não
# tem nenhuma. Contamos quantas bolhas aparecem na região das questões.
PAGINA_MIN_BOLHAS_OBJETIVA = 60
