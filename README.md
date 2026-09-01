# OMR Backend — Leitor do Cartão-Resposta Veloz

Backend em Python + OpenCV que lê, **a partir de uma foto**, as marcações das
folhas *Cartão-Resposta Veloz* (Editora Veloz).

**Dois fluxos**, um por página, cada um no seu endpoint e cada um recebendo a
**folha inteira** fotografada:

| Fluxo | Endpoint | O que devolve |
|---|---|---|
| Objetiva | `POST /anos-iniciais/omr/objetiva` | número do aluno + Linguagens + Matemática |
| Redação | `POST /anos-iniciais/omr/redacao` | número do aluno + quadro de correção (situação + competências 01–05) |

**Dois modelos de folha**, reconhecidos automaticamente — o modelo detectado
volta no campo `model` da resposta:

| Modelo | Linguagens | Matemática | Redação |
|---|---|---|---|
| `anos_iniciais` | 1–21 | 1–22 | — |
| `anos_finais` | 1–25 | 1–26 | sim |

A perspectiva é corrigida pelos **4 marcadores fiduciais** impressos nos cantos
da folha, então a foto pode estar torta, inclinada ou tirada de lado.

## Status por campo

| Situação na folha | Status |
|---|---|
| Exatamente uma bolha marcada | `OK` (com `answer` / `value`) |
| Nenhuma marcada | `BLANK` |
| Duas ou mais marcadas | `MULTIPLE` |
| Preenchimento ambíguo (borrão, rasura) | `REVIEW` — conferir a olho |

## Como funciona

```
foto  ─▶ 1. acha o papel na cena (maior quadrilátero) e retifica grosseiro
      ─▶ 2. acha as 4 marcas fiduciais por template matching multiescala
      ─▶ 3. homografia → "folha canônica" (1700 px de largura entre fiduciais)
      ─▶ 4. identifica o MODELO pela altura das linhas de questão
      ─▶ 5. por bloco: reancora a grade nas bolhas realmente impressas
      ─▶ 6. mede o % de preenchimento de cada bolha e classifica
```

### Por que o passo 4 existe

Os dois modelos compartilham o passo entre linhas. Ler um Anos Iniciais com o
template de Anos Finais **não falha ruidosamente**: as grades se interpenetram e
o resultado são respostas deslocadas uma questão — medido antes da correção,
uma folha de Iniciais lida como Finais devolvia 37 respostas "OK", todas erradas.

Por isso o modelo é decidido pela geometria **medida** das linhas de bolha
(onde o bloco começa, onde termina, quantas linhas tem), e não pela cobertura da
grade. Nas folhas reais o custo de encaixe fica em 0,01 para o modelo certo e
~4,0 para o errado — margem de 400×. Empate ou custo alto → o endpoint recusa,
porque errar o modelo é pior do que não responder.

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
| `POST` | `/anos-iniciais/omr/objetiva` | foto da página 1 → JSON com nº do aluno e as respostas |
| `POST` | `/anos-iniciais/omr/objetiva/debug` | o mesmo, devolvendo a **foto anotada** (PNG) |
| `POST` | `/anos-iniciais/omr/redacao` | foto da página 2 → JSON com nº do aluno e a correção |
| `POST` | `/anos-iniciais/omr/redacao/debug` | o mesmo, devolvendo a **foto anotada** (PNG) |
| `GET` | `/docs` | Swagger UI — dá para testar pelo navegador |

No Postman: `POST http://localhost:8000/anos-iniciais/omr/objetiva` → aba **Body** →
**form-data** → key `file` (troque o tipo de **Text** para **File**) → escolha a
foto → **Send**. Use a URL **sem barra no final**: `/anos-iniciais/omr/objetiva/` responde
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

### Resposta — `/anos-iniciais/omr/objetiva`

```jsonc
{~
  "filename": "folha.jpg",
  "flow": "objetiva",
  "model": "anos_finais",          // ou "anos_iniciais" — detectado
  "model_title": "Cartão-Resposta Veloz — Anos Finais",
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

### Resposta — `/anos-iniciais/omr/redacao`

```jsonc
{
  "filename": "redacao.jpg",
  "flow": "redacao",
  "model": "anos_finais",
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
  girada (`rotation`, em graus), o modo de alinhamento de cada bloco e, em
  `model_detection`, o custo de encaixe de cada modelo. Cobertura baixa é o
  sinal de "peça outra foto".

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
.venv/bin/python -m pytest -q          # 52 testes
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

Cobertura atual: as 6 fotos sintéticas de Anos Finais (fácil/médio/difícil ×
2 fluxos) leem **100 % dos campos**. Nas folhas reais digitalizadas, a de Anos
Iniciais lê **43/43 questões** e o número do aluno inteiro — e ela é o caso
difícil, porque o aluno **circulou** as bolhas em vez de pintar. Num teste de estresse com 30 fotos aleatórias (perspectiva
até 5,5 %, sombra até 55 %, JPEG até qualidade 62), **855 campos comparados, 0
divergência**.

## Calibração / manutenção do template

Tudo o que depende do desenho da folha está em
[omr/template.py](omr/template.py); os limiares e parâmetros de imagem estão em
[omr/config.py](omr/config.py).

Se a Editora reimprimir a folha — ou para acrescentar um modelo novo:

```bash
# rasteriza UMA página do PDF oficial, já endireitada, e mede
.venv/bin/python tools/calibrar_template.py --render --pdf "folha.pdf" \
    --modelo anos_iniciais --fluxo objetiva --pagina 1 --rotacao 180

# só mede o que já está rasterizado e confere contra o template
.venv/bin/python tools/calibrar_template.py --conferir
```

`--pagina` e `--rotacao` existem porque o PDF da coleção traz várias páginas e
algumas vêm giradas 180°; a rotação é aplicada **antes** de medir. As caixas de
busca saem do próprio `template.py`, então o calibrador acompanha um modelo novo
sem lista de coordenadas escrita à mão.

A ferramenta mede os fiduciais e todos os círculos impressos, ajusta uma grade a
cada bloco e imprime as constantes prontas para colar no template — junto com o
resíduo do ajuste. O teste `test_template_bate_com_o_pdf` roda essa mesma
comparação, então um drift silencioso quebra a suíte.

Os limiares de decisão (`MARK_THRESHOLD = 55`, `REVIEW_LOW = 40`) têm folga
medida: bolha vazia fica abaixo de ~26 % e bolha marcada acima de ~82 %.

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
