# Plan 001: Substituir a grade fixa por detecção de grade ancorada nas bolhas reais (+ normalização de iluminação)

> **Instruções ao executor**: Siga este plano passo a passo. Rode TODOS os
> comandos de verificação e confirme a saída esperada antes de avançar. Se
> ocorrer qualquer item da seção "Condições de PARADA", pare e reporte — não
> improvise. Ao terminar, atualize a linha de status deste plano em
> `plans/README.md`.
>
> **Drift check (rode primeiro)**: este projeto NÃO usa git. Em vez de comparar
> por SHA, abra `omr/engine.py` e `omr/config.py` e confirme que os excertos da
> seção "Estado atual" batem com o código vivo. Se não baterem, trate como
> condição de PARADA.

## Status

- **Prioridade**: P1
- **Esforço**: M
- **Risco**: MÉDIO
- **Depende de**: nenhum
- **Categoria**: bug (correção de raiz)
- **Planejado em**: sem VCS. Validado contra `samples/gabarito_matematica.png`
  e `test_images/foto_real.jpg` em 2026-07-27.
- **Ambiente**: venv em `.venv/` (Python 3.12). Rode tudo com `.venv/bin/python`.

## Por que isso importa

O motor lê corretamente um **scan achatado** (a folha de referência), mas erra
quase tudo numa **foto de celular real**: numa foto de teste com 16 marcações,
o retorno teve ~2 questões certas — linhas de cima leram `100%` em todas as
alternativas e a coluna D leu `0%` em quase toda questão.

Causa raiz **medida** (não suposta): a amostragem usa coordenadas ABSOLUTAS e
fixas (`COL_X`/`ROW_Y` em `omr/config.py`), calibradas num scan reto. Numa foto
com inclinação de perspectiva, as bolhas reais, após a retificação, caem em
posições diferentes das que o template assume. Medição no espaço canônico de
400px de altura:

| | Bolhas reais (detectadas) | Template procura em |
|---|---|---|
| Scan de referência | X:[81–241] Y:[32–360] | X:[82–240] Y:[33–359] ✅ |
| Foto de celular | X:[93–219] Y:[**131**–362] | X:[82–240] Y:[33–359] ❌ |

A 1ª linha de bolhas cai em Y=131 mas o template amostra em Y=33 (≈100px acima,
em cima da borda do box → lê 100%); as bolhas terminam em X=219 mas a coluna D
é amostrada em X=240 (no branco à direita → lê 0%).

**A solução (validada neste plano) detecta onde as bolhas realmente estão** e
ancora a amostragem nelas, em vez de confiar em coordenadas fixas. Com isso, o
mesmo motor passa a ler **16/16** tanto no scan quanto na foto de celular, sem
regressão. Como bônus, uma normalização de iluminação (flat-field) elimina o
falso-escuro causado por sombra irregular na foto.

> Este plano corrige o **alinhamento** (a causa raiz). Ele NÃO cobre: robustez
> da detecção do box em si (dedo/curvatura), limiar relativo por questão, nem
> redesenho da folha com marcadores fiduciais. Esses são achados separados
> listados em `plans/README.md` como follow-ups.

## Estado atual

Arquivos envolvidos:

- `omr/config.py` — template + limiares. As constantes `COL_X`, `ROW_Y`,
  `CANON_W=280`, `CANON_H=400`, `SAMPLE_RADIUS=9` definem a grade RÍGIDA que
  falha. Excerto atual (`omr/config.py:13-40`):

```python
CANON_W = 280
CANON_H = 400
COL_X = [82.5, 135.8, 188.2, 240.5]
ROW_Y = [33.3, 79.2, 125.1, 171.3, 217.9, 264.9, 311.9, 359.1]
CHOICES = ["A", "B", "C", "D"]
SAMPLE_RADIUS = 9
QUESTIONS_PER_BOX = 8
NUM_BOXES = 2
MARK_THRESHOLD = 60.0
REVIEW_LOW = 40.0
```

- `omr/engine.py` — pipeline. Dois pontos que este plano altera:

  1. `_warp` (`omr/engine.py:80-83`) — retifica o box para `CANON_W×CANON_H`:
  ```python
  def _warp(gray: np.ndarray, quad: np.ndarray) -> np.ndarray:
      dst = np.array([[0, 0], [C.CANON_W, 0], [C.CANON_W, C.CANON_H], [0, C.CANON_H]], "float32")
      M = cv2.getPerspectiveTransform(quad, dst)
      return cv2.warpPerspective(gray, M, (C.CANON_W, C.CANON_H))
  ```

  2. O laço de amostragem em `process_image` (`omr/engine.py:143-172`) — usa a
     grade fixa e um Otsu global:
  ```python
  for quad in boxes:
      warp = _warp(gray, quad)
      binary = cv2.threshold(
          cv2.GaussianBlur(warp, (3, 3), 0), 0, 255,
          cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU,
      )[1]
      inv_M = None
      if draw_debug:
          dst = np.array([[0, 0], [C.CANON_W, 0], [C.CANON_W, C.CANON_H], [0, C.CANON_H]], "float32")
          inv_M = cv2.getPerspectiveTransform(dst, quad)
      for ry in C.ROW_Y:
          fills = {}
          for ch, cx in zip(C.CHOICES, C.COL_X):
              fills[ch] = round(_bubble_fill(binary, cx, ry, C.SAMPLE_RADIUS), 1)
          answer, status = _classify(fills)
          ...
          if draw_debug:
              _draw_row(debug_img, inv_M, ry, fills, status)
          q += 1
  ```

- `_bubble_fill` (`omr/engine.py:89-99`), `_classify` (`omr/engine.py:102-113`)
  e `_draw_row` (`omr/engine.py:186-201`) — este plano ajusta `_draw_row` e o
  chamador de `_bubble_fill`; `_classify` fica igual.

Convenções do repositório a seguir:
- Comentários e docstrings em português (veja `omr/engine.py`).
- Funções internas com prefixo `_`; a API pública é `process_image`,
  `read_image_file`, `decode_and_process` (não altere as assinaturas delas —
  `app.py` e `test_cli.py` dependem do formato de retorno atual).
- O dicionário de retorno por questão (`question`, `answer`, `status`,
  `marked`, `fills`) **não muda de formato** — apenas passa a vir correto.

## Comandos que você vai usar

| Propósito | Comando | Esperado |
|---|---|---|
| Ler o scan de referência | `.venv/bin/python test_cli.py samples/gabarito_matematica.png` | 16 linhas, todas `OK` |
| Ler a foto de celular | `.venv/bin/python test_cli.py test_images/foto_real.jpg` | Q1 `MULTIPLA`, Q2–16 `OK` |
| Debug visual | `.venv/bin/python test_cli.py test_images/foto_real.jpg --debug` | gera `debug/foto_real_debug.png` |
| Testes | `.venv/bin/pip install pytest && .venv/bin/python -m pytest -q` | todos passam |

> Se `test_images/foto_real.jpg` não existir, é uma condição de PARADA (o
> conjunto de teste é pré-requisito — veja Plano 000 no README). Não invente
> uma imagem.

## A solução — código de referência VALIDADO

As funções abaixo já foram testadas contra o scan de referência (16/16) **e** a
foto de celular real (16/16, com Q1 corretamente `MULTIPLE`). Adapte-as ao
estilo do arquivo, mas **não altere a lógica nem as constantes numéricas** —
elas foram calibradas empiricamente.

```python
# Tamanho canônico maior preserva resolução das bolhas para a detecção.
# (substitui CANON_W=280/CANON_H=400)
GRID_W, GRID_H = 800, 1143          # aspecto ~0.70 do box
FLATFIELD_SIGMA = 41                # blur do flat-field p/ corrigir sombra
BUBBLE_R_EXPECTED = 27              # raio esperado da bolha no canônico 800-wide
ADAPT_BLOCK, ADAPT_C = 41, 8        # adaptiveThreshold p/ detectar candidatas
SAMPLE_R_FRAC = 0.72               # raio de amostragem = fração do raio detectado

def _normalize_illumination(warp):
    """Flat-field: divide a imagem por uma versão muito borrada -> remove
    gradiente de sombra (persiana, iluminação lateral)."""
    bg = cv2.GaussianBlur(warp, (0, 0), FLATFIELD_SIGMA)
    return cv2.divide(warp, bg, scale=255).astype("uint8")

def _split_k(vals, k):
    """Agrupa valores 1D em k grupos cortando nas (k-1) maiores lacunas.
    Devolve as medianas dos grupos, ordenadas. None se houver < k valores."""
    v = np.sort(np.asarray(vals, dtype=float))
    if len(v) < k:
        return None
    cut = np.sort(np.argsort(np.diff(v))[-(k - 1):])
    return np.array([np.median(g) for g in np.split(v, cut + 1)])

def _linfit(centers):
    """Regulariza para espaçamento uniforme: ajusta center[i]=a+b*i."""
    idx = np.arange(len(centers))
    b, a = np.polyfit(idx, centers, 1)
    return a + b * idx

def _detect_grid(warp):
    """Detecta as posições REAIS das 4 colunas e 8 linhas de bolhas dentro do
    box já retificado. Devolve (cols[4], rows[8], raio_mediano).
    Levanta OMRError se a grade não puder ser recuperada com confiança."""
    norm = _normalize_illumination(warp)
    th = cv2.adaptiveThreshold(norm, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, ADAPT_BLOCK, ADAPT_C)
    cnts, _ = cv2.findContours(th, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cand = []
    for c in cnts:
        a = cv2.contourArea(c)
        if a < 300:
            continue
        (cx, cy), r = cv2.minEnclosingCircle(c)
        if not (0.66 * BUBBLE_R_EXPECTED <= r <= 1.55 * BUBBLE_R_EXPECTED):
            continue
        if a / (np.pi * r * r) < 0.45:       # descarta formas não-circulares
            continue
        cand.append((cx, cy, r))
    if len(cand) < 20:
        raise OMRError("Poucas bolhas detectadas na coluna; foto possivelmente "
                       "fora de foco, cortada ou com contraste ruim.")
    colc = _split_k([p[0] for p in cand], 4)
    rowc = _split_k([p[1] for p in cand], 8)
    if colc is None or rowc is None:
        raise OMRError("Não foi possível separar 4 colunas / 8 linhas de bolhas.")
    cols, rows = _linfit(colc), _linfit(rowc)
    # sanidade: espaçamento regular (resíduo do ajuste linear pequeno)
    col_pitch = np.mean(np.diff(cols)); row_pitch = np.mean(np.diff(rows))
    if col_pitch <= 0 or row_pitch <= 0:
        raise OMRError("Grade degenerada (pitch inválido).")
    if (np.max(np.abs(colc - cols)) > 0.30 * col_pitch or
            np.max(np.abs(rowc - rows)) > 0.30 * row_pitch):
        raise OMRError("Grade irregular; alinhamento não confiável — refaça a foto.")
    rmed = float(np.median([p[2] for p in cand]))
    return cols, rows, rmed

def _measure_fills(warp, cols, rows, rmed):
    """% de preenchimento de cada bolha nas posições DETECTADAS. Devolve uma
    lista de 8 linhas, cada uma com 4 valores (A,B,C,D)."""
    binimg = cv2.threshold(_normalize_illumination(warp), 0, 255,
                           cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    rs = int(round(SAMPLE_R_FRAC * rmed))
    mask = np.zeros((2 * rs, 2 * rs), np.uint8)
    cv2.circle(mask, (rs, rs), rs, 255, -1)
    area = cv2.countNonZero(mask)
    grid = []
    for ry in rows:
        row = []
        for cx in cols:
            x0, y0 = int(cx - rs), int(ry - rs)
            patch = binimg[y0:y0 + 2 * rs, x0:x0 + 2 * rs]
            if patch.shape[:2] != (2 * rs, 2 * rs):
                row.append(0.0)
            else:
                row.append(100.0 * cv2.countNonZero(
                    cv2.bitwise_and(patch, patch, mask=mask)) / area)
        grid.append(row)
    return grid
```

Separação medida com este código (confirma que `MARK_THRESHOLD` continua seguro):
- Foto de celular: bolha marcada 82–99%, bolha vazia 10–26% (vão de ~56 pontos).
- Scan de referência: marcada ~100%, vazia ≤21%.

## Escopo

**No escopo** (só estes arquivos):
- `omr/config.py` — trocar constantes de grade fixa por parâmetros de detecção.
- `omr/engine.py` — adicionar helpers acima; reescrever o laço de amostragem;
  ajustar `_draw_row`.
- `tests/test_reading.py` (criar) — ver Plano de teste.
- `requirements.txt` — acrescentar `pytest` (dev).

**Fora do escopo** (NÃO toque, mesmo parecendo relacionado):
- `_find_boxes` (`omr/engine.py:44-77`) — robustez da detecção do box é o
  achado #4, plano separado. Ela já entrega os 2 boxes nesta foto.
- `_classify` (`omr/engine.py:102-113`) — a lógica branco/única/múltipla/revisar
  está correta; só passa a receber `fills` corretos.
- `app.py`, `test_cli.py` — as interfaces e o formato de retorno não mudam.
- Limiar relativo por questão (achado #3) e marcadores fiduciais (caminho B).

## Passos

### Passo 1: Trocar as constantes de grade em `omr/config.py`

Remova `CANON_W`, `CANON_H`, `COL_X`, `ROW_Y`, `SAMPLE_RADIUS` (a grade fixa) e
adicione os parâmetros de detecção. Mantenha `CHOICES`, `QUESTIONS_PER_BOX`,
`NUM_BOXES`, `MARK_THRESHOLD`, `REVIEW_LOW`, `BOX_MIN_AREA_FRAC`,
`BOX_MAX_AREA_FRAC`. Adicione:

```python
# Grade é DETECTADA, não fixa (ver omr/engine._detect_grid).
GRID_W = 800          # largura canônica do box retificado
GRID_H = 1143         # altura (aspecto ~0.70)
FLATFIELD_SIGMA = 41
BUBBLE_R_EXPECTED = 27
ADAPT_BLOCK = 41
ADAPT_C = 8
SAMPLE_R_FRAC = 0.72
```

Ajuste `MARK_THRESHOLD = 55.0` (marcas rabiscadas leem a partir de ~82%; 55 dá
margem sem risco). Deixe `REVIEW_LOW = 40.0`.

**Verify**: `.venv/bin/python -c "from omr import config as c; print(c.GRID_W, c.MARK_THRESHOLD)"` → `800 55.0`

### Passo 2: Adicionar os helpers em `omr/engine.py`

Cole as funções `_normalize_illumination`, `_split_k`, `_linfit`,
`_detect_grid`, `_measure_fills` (código de referência acima) logo após
`_warp`. Use `C.GRID_W`, `C.GRID_H`, `C.FLATFIELD_SIGMA`, etc. em vez dos
literais (substitua `GRID_W`→`C.GRID_W`, `FLATFIELD_SIGMA`→`C.FLATFIELD_SIGMA`,
`BUBBLE_R_EXPECTED`→`C.BUBBLE_R_EXPECTED`, `ADAPT_BLOCK`→`C.ADAPT_BLOCK`,
`ADAPT_C`→`C.ADAPT_C`, `SAMPLE_R_FRAC`→`C.SAMPLE_R_FRAC`).

Atualize `_warp` para o novo tamanho canônico:
```python
def _warp(gray: np.ndarray, quad: np.ndarray) -> np.ndarray:
    dst = np.array([[0, 0], [C.GRID_W, 0], [C.GRID_W, C.GRID_H], [0, C.GRID_H]], "float32")
    M = cv2.getPerspectiveTransform(quad, dst)
    return cv2.warpPerspective(gray, M, (C.GRID_W, C.GRID_H))
```

**Verify**: `.venv/bin/python -c "import omr.engine"` → sem erro de import.

### Passo 3: Reescrever o laço de amostragem em `process_image`

Substitua o corpo do `for quad in boxes:` (linhas ~143-172) por: warp →
detectar grade → medir preenchimentos nas posições detectadas → classificar.
Guarde `cols`, `rows`, `rmed` para o debug. Alvo:

```python
for quad in boxes:
    warp = _warp(gray, quad)
    cols, rows, rmed = _detect_grid(warp)
    grid = _measure_fills(warp, cols, rows, rmed)

    inv_M = None
    if draw_debug:
        dst = np.array([[0, 0], [C.GRID_W, 0], [C.GRID_W, C.GRID_H], [0, C.GRID_H]], "float32")
        inv_M = cv2.getPerspectiveTransform(dst, quad)

    for ri, ry in enumerate(rows):
        fills = {ch: round(grid[ri][ci], 1) for ci, ch in enumerate(C.CHOICES)}
        answer, status = _classify(fills)
        results.append({
            "question": q, "answer": answer, "status": status,
            "marked": [ch for ch, f in fills.items() if f >= C.MARK_THRESHOLD],
            "fills": fills,
        })
        if draw_debug:
            _draw_row(debug_img, inv_M, cols, ry, rmed, fills, status)
        q += 1
```

Note que `_draw_row` ganha os parâmetros `cols`, `ry`, `rmed` (o desenho agora
usa as posições detectadas). `_bubble_fill` deixa de ser chamado aqui (a medição
migrou para `_measure_fills`); pode mantê-lo no arquivo (inofensivo) ou removê-lo.

**Verify**: `.venv/bin/python test_cli.py samples/gabarito_matematica.png` →
16 linhas, todas com status `OK` e respostas `D B A A D D B C C B B B A B C D`.

### Passo 4: Atualizar `_draw_row` para desenhar nas posições detectadas

Nova assinatura e corpo (desenha um círculo por coluna detectada + a tag de
status; raio proporcional ao detectado):

```python
def _draw_row(img, inv_M, cols, ry, rmed, fills, status):
    color_status = {
        "OK": (0, 170, 0), "BLANK": (0, 200, 200),
        "MULTIPLE": (0, 0, 230), "REVIEW": (0, 140, 255),
    }[status]
    for ch, cx in zip(C.CHOICES, cols):
        pt = cv2.perspectiveTransform(np.array([[[cx, ry]]], "float32"), inv_M)[0][0]
        marked = fills[ch] >= C.MARK_THRESHOLD
        col = (0, 170, 0) if marked else (150, 150, 150)
        cv2.circle(img, (int(pt[0]), int(pt[1])), int(rmed * 1.2), col, 4 if marked else 2)
    p0 = cv2.perspectiveTransform(np.array([[[cols[0] - 2 * rmed, ry]]], "float32"), inv_M)[0][0]
    cv2.putText(img, status[:4], (int(p0[0]), int(p0[1])),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color_status, 2, cv2.LINE_AA)
```

**Verify**: `.venv/bin/python test_cli.py test_images/foto_real.jpg --debug` →
gera `debug/foto_real_debug.png`; a saída no terminal mostra Q1 `MULTIPLA` e
Q2–16 `OK`. Abra o PNG e confirme que os círculos verdes caem SOBRE as bolhas
marcadas.

### Passo 5: Rodar a foto de celular e confirmar o resultado correto

**Verify**: `.venv/bin/python test_cli.py test_images/foto_real.jpg` deve
produzir exatamente:

```
Q 1  MULTIPLA   resp=-
Q 2  OK         resp=B
Q 3  OK         resp=A
Q 4  OK         resp=A
Q 5  OK         resp=D
Q 6  OK         resp=D
Q 7  OK         resp=B
Q 8  OK         resp=C
Q 9  OK         resp=C
Q10  OK         resp=B
Q11  OK         resp=B
Q12  OK         resp=B
Q13  OK         resp=A
Q14  OK         resp=B
Q15  OK         resp=C
Q16  OK         resp=D
```

(Q1 é `MULTIPLA` porque a folha tem A **e** D preenchidos — é o comportamento
correto, não um erro.)

## Plano de teste

Crie `tests/test_reading.py` (pytest). Cubra: leitura correta do scan, leitura
correta da foto real, detecção de dupla marcação, e não-regressão dos sintéticos
branco/múltipla (gerados por `make_test_images.py`, se presentes).

```python
import os, pytest
from omr import engine

ROOT = os.path.dirname(os.path.dirname(__file__))

def _answers(path):
    res = engine.read_image_file(os.path.join(ROOT, path))
    return {r["question"]: (r["status"], r["answer"]) for r in res["results"]}

def test_scan_referencia_16_de_16():
    a = _answers("samples/gabarito_matematica.png")
    esperado = list("DBAADDBC" "CBBBABCD")
    for q, letra in enumerate(esperado, start=1):
        assert a[q] == ("OK", letra), f"Q{q} veio {a[q]}"

def test_foto_celular_real():
    a = _answers("test_images/foto_real.jpg")
    assert a[1][0] == "MULTIPLE"                      # A e D preenchidos
    esperado = {2:"B",3:"A",4:"A",5:"D",6:"D",7:"B",8:"C",
                9:"C",10:"B",11:"B",12:"B",13:"A",14:"B",15:"C",16:"D"}
    for q, letra in esperado.items():
        assert a[q] == ("OK", letra), f"Q{q} veio {a[q]}"
```

**Verify**: `.venv/bin/pip install pytest && .venv/bin/python -m pytest -q` →
2 passed (ou mais, se adicionar os sintéticos).

## Critérios de conclusão

Todos devem valer:

- [ ] `.venv/bin/python test_cli.py samples/gabarito_matematica.png` → 16 `OK`, respostas `D B A A D D B C C B B B A B C D`.
- [ ] `.venv/bin/python test_cli.py test_images/foto_real.jpg` → Q1 `MULTIPLA`, Q2–16 `OK` com as letras do Passo 5.
- [ ] `.venv/bin/python -m pytest -q` → todos passam.
- [ ] `debug/foto_real_debug.png` mostra os círculos ancorados nas bolhas reais.
- [ ] Nenhum arquivo fora do escopo foi alterado.
- [ ] Linha de status atualizada em `plans/README.md`.

## Condições de PARADA

Pare e reporte (não improvise) se:

- Os excertos da seção "Estado atual" não baterem com o código vivo (o código
  mudou desde este plano).
- `test_images/foto_real.jpg` não existir.
- `_detect_grid` levantar `OMRError` no scan de referência ou na foto real
  (nesses dois casos a grade DEVE ser detectável; falha indica que algum
  parâmetro foi copiado errado — revise `BUBBLE_R_EXPECTED`, `GRID_W/H`).
- Qualquer verificação falhar duas vezes após uma tentativa razoável de ajuste.
- A correção parecer exigir mexer em `_find_boxes` ou em `app.py`/`test_cli.py`
  (é sinal de que o escopo está sendo violado).

## Notas de manutenção

- **Interação futura**: se o modelo da folha mudar o número de questões/colunas,
  ajuste `QUESTIONS_PER_BOX`/`NUM_BOXES` e os `k` em `_split_k` (4 colunas /
  8 linhas estão embutidos em `_detect_grid`).
- `_detect_grid` assume que as bolhas têm contorno detectável (círculo impresso
  ou preenchimento). Se uma folha futura remover o círculo impresso, revise a
  faixa de raio e a circularidade.
- **O que o revisor deve escrutinar**: rodar em 5–10 fotos NOVAS (ângulos e luz
  variados) e olhar os `debug/*.png` — a validação atual cobre 1 scan + 1 foto;
  a robustez real só se confirma com mais amostras (é o achado #5, conjunto de
  teste). Casos que caírem em `OMRError` de grade irregular são o comportamento
  desejado (rejeitar > adivinhar), mas conte quantos, para calibrar o limiar de
  0.30·pitch.
- **Follow-ups deferidos** (não faça aqui): achado #2 já resolvido de fato pela
  normalização (mantido junto por acoplamento); achado #3 (limiar relativo por
  questão) e #4 (detecção de box robusta) continuam abertos; caminho B
  (marcadores fiduciais) é um plano de design à parte.
