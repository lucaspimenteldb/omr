# OMR Backend — Leitor do Cartão-Resposta Veloz (Anos Finais)

Backend em Python + OpenCV que lê, **a partir de uma foto**, as marcações da
folha *Cartão-Resposta Veloz — Anos Finais* (Editora Veloz).

A folha tem duas páginas e o backend expõe **um fluxo por página**, cada um no
seu endpoint, e cada um recebendo a **folha inteira** fotografada:

| Fluxo | Endpoint | Página | O que devolve |
|---|---|---|---|
| Objetiva | `POST /anos-finais/omr/objetiva` | 1 — *CARTÃO-RESPOSTA* | número do aluno + Linguagens (1–25) + Matemática (1–26) |
| Redação | `POST /anos-finais/omr/redacao` | 2 — *PRODUÇÃO DE TEXTO* | número do aluno + quadro de correção (situação + competências 01–05) |

A perspectiva é corrigida pelos **4 marcadores fiduciais** impressos nos cantos
da folha, então a foto pode estar torta, inclinada ou tirada de lado.

## Status por campo

| Situação na folha | Status |
|---|---|
| Exatamente uma bolha marcada | `OK` (com `answer` / `value`) |
| Nenhuma marcada | `BLANK` |
| Duas ou mais marcadas | `MULTIPLE` |
| Preenchimento ambíguo (borrão, rasura, empate técnico) | `REVIEW` — conferir a olho |

### Como a decisão é tomada

Duas regras, **nesta ordem** (ver `_classificar` em [omr/engine.py](omr/engine.py)):

1. **Absoluta** — bolha com preenchimento ≥ `MARK_THRESHOLD` (55 %) conta como
   marcada. É a única regra que enxerga **marcação dupla**, e por isso vem
   primeiro: se duas bolhas foram mesmo pintadas, o campo é `MULTIPLE` e nenhum
   critério relativo pode escolher uma delas.
2. **Relativa** — se *nenhuma* bolha cruzou os 55 %, a mais preenchida vence,
   desde que passe do piso **e** esteja à frente da segunda colocada por
   `VANTAGEM_RELATIVA` (12 pontos). É o que salva a marca fraca: um traço leve
   de 40 % contra vizinhas de 15 % é uma resposta, não uma folha em branco.

Se todas ficarem abaixo do piso → `BLANK`. Se a vencedora não abrir vantagem
sobre a segunda → `REVIEW`, porque escolher aí seria cara-ou-coroa.

O piso é **por bloco** (`pisos_do_bloco` em [omr/config.py](omr/config.py)):

| Bloco | Piso | Fallback |
|---|---|---|
| Questões e quadro de correção | 30 % | — |
| Número do aluno | 40 % | 30 % |

O número do aluno é mais rígido porque um dígito errado troca a identidade da
prova inteira, e a coluna tem 10 opções (mais chance de vizinha suja) contra as
4 de uma questão.

> **Nota honesta sobre os dois degraus.** Como a exigência de vantagem vale nos
> dois e as faixas `[40, ∞)` e `[30, 40)` se encostam, o par 40/30 decide hoje
> **exatamente igual** a um piso único de 30 — o teste
> `test_degraus_do_numero_aluno_sao_contiguos` fixa esse fato. O 40 documenta a
> intenção e deixa o botão pronto: para apertar de verdade, suba
> `PISO_FALLBACK_POR_BLOCO` acima de 30 ou remova o fallback.

## Como funciona

```
foto  ─▶ 1. acha o papel na cena (maior quadrilátero) e retifica grosseiro
      ─▶ 2. acha as 4 marcas fiduciais por template matching multiescala
      ─▶ 3. homografia → "folha canônica" (1700 px de largura entre fiduciais)
      ─▶ 4. por bloco: reancora a grade nas bolhas realmente impressas
      ─▶ 5. mede o % de preenchimento de cada bolha e classifica
```

Cada etapa tolera a falha da anterior: se o papel não for identificável a foto
inteira é tratada como a folha; se só 3 marcas forem achadas, a 4ª é inferida
por paralelogramo; se nenhuma for achada, cai para os cantos do papel. O erro
residual que sobrar é absorvido no passo 4, que **reancora a grade nas bolhas
reais** (ajuste afim global + deslocamento local por bloco) em vez de confiar
cegamente em coordenadas fixas.

A geometria da folha mora em [omr/template.py](omr/template.py), em coordenadas
normalizadas pelo retângulo dos 4 fiduciais — independente de DPI, de tamanho da
foto e de quanta margem de papel apareceu. Os números foram **medidos** no PDF
oficial: o resíduo do ajuste de grade é < 0,1 px, ou seja, a folha é uma grade
perfeitamente regular.

## Instalação

```bash
cd omr-backend
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt          # produção
.venv/bin/pip install -r requirements-dev.txt      # + testes e calibração
```

## Uso 1 — API (Postman / app)

```bash
.venv/bin/uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

| Método | Rota | Descrição |
|---|---|---|
| `GET` | `/health` | status do serviço e o que cada fluxo devolve |
| `POST` | `/anos-finais/omr/objetiva` | foto da página 1 → JSON com nº do aluno e as respostas |
| `POST` | `/anos-finais/omr/objetiva/debug` | o mesmo, devolvendo a **foto anotada** (PNG) |
| `POST` | `/anos-finais/omr/redacao` | foto da página 2 → JSON com nº do aluno e a correção |
| `POST` | `/anos-finais/omr/redacao/debug` | o mesmo, devolvendo a **foto anotada** (PNG) |
| `GET` | `/docs` | Swagger UI — dá para testar pelo navegador |

No Postman: `POST http://localhost:8000/anos-finais/omr/objetiva` → aba **Body** →
**form-data** → key `file` (troque o tipo de **Text** para **File**) → escolha a
foto → **Send**. Use a URL **sem barra no final**: `/anos-finais/omr/objetiva/` responde
`307` e nem todo cliente reenvia o corpo no redirect.

> **Não defina o header `Content-Type` na mão.** Quem monta o `boundary` é o
> cliente; um header manual acaba discordando do corpo e o parser devolve
> `400 Expected boundary character 45, got 148 at index 2`. Em JavaScript, monte
> um `FormData` e passe direto — sem headers. Se cair nesse erro, o `400` da API
> explica o conserto e traz a mensagem crua do parser no campo `parser`.

> Mandou a página errada? O endpoint recusa com `422` e diz qual usar, em vez de
> devolver números sem sentido.

### Formatos aceitos

Mande o arquivo **como a câmera gravou** — inclusive **HEIC**, o padrão do
iPhone. Também entram JPEG, PNG, WEBP, AVIF, TIFF (até 16 bits), BMP, GIF e
JPEG 2000. Não converta antes: a foto é decodificada direto para a matriz de
pixels, sem passar por arquivo intermediário, então não há perda nenhuma no
caminho — reexportar para JPEG, isso sim, custaria qualidade na borda da bolha,
que é exatamente o que o motor mede.

PDF **não** é aceito (é documento, não imagem): exporte a página como foto. O
`422` diz isso com todas as letras em vez do genérico "formato não suportado".

### Resposta — `/anos-finais/omr/objetiva`

```jsonc
{~
  "filename": "folha.jpg",
  "flow": "objetiva",
  "student_number": {
    "value": "1207",              // null se não der para afirmar
    "raw": "______1207",          // 10 casas: dígito, "_" em branco, "?" ambígua
    "status": "OK",
    "digits": [ { "position": 7, "digit": "1", "status": "OK",
                  "marked": ["1"], "fills": { "0": 11.2, "1": 99.8 } } ]
  },
  "sections": [
    { "name": "linguagens", "num_questions": 25,
      "summary": { "ok": 24, "blank": 1, "multiple": 0, "review": 0 },
      "results": [ { "question": 1, "answer": "D", "status": "OK",
                     "marked": ["D"],
                     "fills": { "A": 12.1, "B": 10.4, "C": 11.7, "D": 99.6 } } ] },
    { "name": "matematica", "num_questions": 26, "summary": {}, "results": [] }
  ],
  "summary": { "ok": 49, "blank": 1, "multiple": 1, "review": 0 },
  "alignment": { "fiducials": "fiduciais", "coverage": 0.98, "rotation": 0 }
}
```

### Resposta — `/anos-finais/omr/redacao`

```jsonc
{
  "filename": "redacao.jpg",
  "flow": "redacao",
  "student_number": { "value": "1207", "raw": "______1207", "status": "OK", "digits": [] },
  "correction": {
    "situacao":       { "value": "C", "status": "OK", "marked": ["C"], "fills": {} },
    "competencia_01": { "value": "3", "status": "OK", "marked": ["3"],
                        "fills": {}, "level": 3 },
    "competencia_02": {}, "competencia_03": {}, "competencia_04": {}, "competencia_05": {}
  },
  "summary": { "ok": 5, "blank": 1, "multiple": 0, "review": 0 },
  "alignment": { "fiducials": "fiduciais", "coverage": 0.97, "rotation": 0 }
}
```

`situacao` é uma letra `A`–`E`; as competências são `"0"`–`"4"` em `value` e o
mesmo valor como inteiro em `level` (`null` quando o status não é `OK`).

### O campo `fills` e o campo `alignment`

- **`fills`** (0–100) é o quanto cada bolha ficou preenchida. É a base de
  conferência: dá para ver exatamente por que cada campo foi classificado assim.
- **`alignment`** conta como o registro correu — de onde vieram os fiduciais,
  quanto da grade foi reconhecida (`coverage`), quanto a foto precisou ser
  girada (`rotation`, em graus) e o modo de alinhamento de cada bloco. Cobertura
  baixa é o sinal de "peça outra foto".

**Orientação é resolvida sozinha.** A folha pode vir em pé, deitada (90°/270°)
ou de ponta-cabeça: o motor testa as quatro orientações e fica com a que
reconhece mais bolhas do template. A foto em pé custa uma tentativa só — as
outras três só rodam se a primeira não fechar.

## Uso 2 — CLI (sem subir a API)

```bash
.venv/bin/python test_cli.py foto.jpg                    # descobre o fluxo sozinho
.venv/bin/python test_cli.py foto.jpg --fluxo redacao    # força o fluxo
.venv/bin/python test_cli.py *.jpg --debug               # salva a foto anotada em debug/
.venv/bin/python test_cli.py foto.jpg --json             # JSON bruto
```

A foto anotada (`--debug` ou `/omr/<fluxo>/debug`) mostra, sobre a sua própria
foto: **verde** = bolha lida como marcada, **cinza** = vazia, **laranja** =
ambígua, e uma **moldura magenta** ligando os 4 fiduciais como o motor os
entendeu. É a forma mais rápida de conferir se a grade caiu no lugar certo.

## Testes

```bash
.venv/bin/python -m pytest -q          # 65 testes
```

Sem acervo de fotos reais anotadas, a validação usa **fotos sintéticas**: o PDF
oficial é renderizado, bolhas são pintadas em posições escolhidas por nós e a
imagem é degradada como uma foto de celular degrada — perspectiva, sombra,
desfoque, ruído e JPEG. Como o gabarito é conhecido por construção, a comparação
é exata.

```bash
.venv/bin/python make_test_images.py   # gera as fotos em test_images/sinteticas/
```

Os testes **não** dependem desses arquivos (geram tudo em memória, com sementes
fixas); o que fica em disco serve para conferir na mão e no Postman.

Cobertura atual: as 6 fotos sintéticas (fácil/médio/difícil × 2 fluxos) leem
**100 % dos campos**, incluindo número do aluno, questões em branco e questões
com marcação dupla. Num teste de estresse com 30 fotos aleatórias (perspectiva
até 5,5 %, sombra até 55 %, JPEG até qualidade 62), **855 campos comparados, 0
divergência**.

## Calibração / manutenção do template

Tudo o que depende do desenho da folha está em
[omr/template.py](omr/template.py); os limiares e parâmetros de imagem estão em
[omr/config.py](omr/config.py).

Se a Editora reimprimir a folha com qualquer mudança de layout:

```bash
# rederiva a geometria do PDF novo e compara com o template atual
.venv/bin/python tools/calibrar_template.py --render --pdf "folha nova.pdf"
```

A ferramenta mede os fiduciais e todos os círculos impressos, ajusta uma grade a
cada bloco e imprime as constantes prontas para colar no template — junto com o
resíduo do ajuste. O teste `test_template_bate_com_o_pdf` roda essa mesma
comparação, então um drift silencioso quebra a suíte.

Os limiares de decisão têm folga medida. A bolha vazia mais escura já observada
(em 4 000+ amostras, sintéticas e reais) ficou em **23,3 %**; bolha bem pintada
fica acima de 82 %.

| Constante | Valor | Folga até o dado medido |
|---|---|---|
| `MARK_THRESHOLD` | 55 % | 27 pontos abaixo da marca mais fraca |
| `PISO_RELATIVO` | 30 % | ~7 pontos acima da vazia mais escura |
| `PISO_RELATIVO_POR_BLOCO` | 40 % (nº do aluno) | ~17 pontos |
| `PISO_FALLBACK_POR_BLOCO` | 30 % (nº do aluno) | ~7 pontos |
| `VANTAGEM_RELATIVA` | 12 pts | entre bolhas todas vazias a diferença típica é < 10 |

O `PISO_RELATIVO` é o limiar mais apertado do sistema — é o preço de capturar
marca fraca. Se aparecerem folhas com fundo sujo ou digitalização de baixo
contraste, é o primeiro número a revisar (e o `fills` da resposta é o dado para
decidir).

## Estrutura

```
omr-backend/
├── omr/
│   ├── template.py       # o modelo da folha em coordenadas normalizadas
│   ├── registration.py   # fiduciais → homografia → folha canônica
│   ├── engine.py         # reancoragem da grade, medição e classificação
│   ├── flows.py          # os 2 fluxos + imagem anotada
│   ├── config.py         # parâmetros de imagem e limiares
│   └── utils.py          # entrada de imagem (bytes / arquivo)
├── app.py                # API FastAPI
├── test_cli.py           # leitura por arquivo, sem subir a API
├── make_test_images.py   # gera as fotos sintéticas de teste
├── tools/
│   └── calibrar_template.py   # rederiva o template a partir do PDF
├── tests/
│   ├── test_leitura.py   # ponta a ponta (motor)
│   └── test_api.py       # contrato dos endpoints
├── samples/modelo/       # PDF oficial + renders 300 dpi (fonte da verdade)
├── samples/legado/       # imagens do modelo ANTIGO de folha (histórico)
└── requirements*.txt
```

## Limitações conhecidas

- **Marca fraca demais pode virar `BLANK`.** A normalização de iluminação
  (flat-field) reduz o contraste de marcas muito claras e grandes. A própria
  folha exige caneta preta ou azul-escura e proíbe lápis — dentro dessa regra a
  margem é grande, mas uma marca a lápis fraco pode ser subestimada. Confira o
  `fills` quando o resultado surpreender.
- **Uma folha por foto.** Duas folhas no mesmo enquadramento levam a detecção do
  papel a escolher uma delas.
- A leitura só olha as bolhas. Nome, escola, ano e turma são escritos à mão e
  **não** são reconhecidos — o `student_number` é a única identificação lida.
