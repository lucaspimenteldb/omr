"""
Configuração / calibração do leitor de gabarito.

Calibrado para a folha de 16 questões, 2 colunas, alternativas A/B/C/D
(samples/gabarito_matematica.png). A grade de bolhas NÃO é mais fixa: ela é
DETECTADA em cada foto (ver omr/engine._detect_grid), o que torna a leitura
robusta a fotos com perspectiva/inclinação. Aqui ficam só os parâmetros da
detecção e os limiares de decisão.
"""

# --- Tamanho canônico de cada box de coluna após a retificação de perspectiva ---
# Maior que o box real preserva resolução das bolhas para a detecção da grade.
GRID_W = 800          # largura canônica do box retificado
GRID_H = 1143         # altura (aspecto ~0.70 do box)

# --- Parâmetros da detecção de bolhas / grade (no espaço canônico acima) ---
FLATFIELD_SIGMA = 41      # sigma do blur do flat-field (corrige sombra irregular)
BUBBLE_R_EXPECTED = 27    # raio esperado da bolha no canônico GRID_W-wide
ADAPT_BLOCK = 41          # blockSize do adaptiveThreshold p/ achar candidatas
ADAPT_C = 8               # constante C do adaptiveThreshold
SAMPLE_R_FRAC = 0.72      # raio de amostragem = fração do raio detectado

# Rótulo de cada coluna (A, B, C, D), na ordem da esquerda para a direita.
CHOICES = ["A", "B", "C", "D"]

# Ordem de leitura dos boxes: da esquerda para a direita.
# O 1º box cobre as questões 1..8, o 2º cobre 9..16, e assim por diante.
QUESTIONS_PER_BOX = 8
NUM_BOXES = 2

# --- Limiares de decisão (percentual de pixels escuros dentro da bolha) ---
# Separação medida (imagem normalizada por flat-field):
#   bolha VAZIA  -> ~8%..26%
#   bolha MARCADA-> ~82%..100%  (inclui marcas rabiscadas, não só preenchidas)
# O vão é grande (~56 pontos), então 55% fica seguro para os dois lados.
MARK_THRESHOLD = 55.0      # fill >= isto  => bolha considerada MARCADA
REVIEW_LOW = 40.0          # fill em [REVIEW_LOW, MARK_THRESHOLD) => AMBÍGUA (revisar)

# --- Detecção dos boxes de coluna na imagem inteira ---
# Cada box deve ocupar entre estes limites da área total da imagem.
BOX_MIN_AREA_FRAC = 0.04
BOX_MAX_AREA_FRAC = 0.45
