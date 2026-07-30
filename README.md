# OMR Backend — Leitor de Gabaritos (OpenCV)

Backend em Python + OpenCV para ler as respostas marcadas por alunos em folhas
de gabarito, a partir de uma **foto**. Calibrado para a folha modelo de
**16 questões, 2 colunas, alternativas A/B/C/D**.

Foco em precisão nos 3 cenários:

| Cenário na folha | Status retornado |
|---|---|
| Nenhuma alternativa marcada | `BLANK` |
| Mais de uma alternativa marcada | `MULTIPLE` |
| Exatamente uma marcada | `OK` (com `answer`) |
| Preenchimento ambíguo (borrão/rasura) | `REVIEW` (conferir manualmente) |

## Como funciona

1. Detecta os 2 boxes de coluna na foto (contornos retangulares grandes).
2. Retifica cada box com *perspective warp* (corrige inclinação/ângulo da foto).
3. Binariza (Otsu) e mede o **% de preenchimento** de cada bolha usando uma
   grade fixa calibrada (`omr/config.py`).
4. Classifica cada questão. Calibração medida na folha real: bolha vazia ≤ 31 %,
   bolha marcada ≥ 97 % → limiar em **60 %** com margem de ~66 pontos.

## Instalação

```bash
cd omr-backend
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Uso 1 — Script CLI (fotos de teste)

```bash
.venv/bin/python test_cli.py samples/gabarito_matematica.png
.venv/bin/python test_cli.py minha_foto.jpg --debug   # salva imagem anotada em debug/
.venv/bin/python test_cli.py *.jpg --json             # saída JSON bruta
```

## Uso 2 — API (Postman) — recomendado

```bash
.venv/bin/uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Endpoints:

| Método | Rota | Descrição |
|---|---|---|
| GET | `/health` | status do serviço |
| POST | `/omr` | recebe a foto (form-data, campo **`file`**) → JSON com as marcações |
| POST | `/omr/debug` | mesma coisa, mas devolve a **imagem anotada** (PNG) para conferência |
| GET | `/docs` | Swagger UI — dá para testar pelo navegador, sem Postman |

### No Postman
1. `POST http://localhost:8000/omr`
2. Aba **Body** → **form-data**
3. Key = `file`, tipo **File**, e selecione a foto do gabarito.
4. **Send**.

### Exemplo de resposta

```json
{
  "filename": "foto.jpg",
  "num_questions": 16,
  "results": [
    {"question": 1, "answer": "D", "status": "OK",
     "marked": ["D"], "fills": {"A":14.7,"B":14.3,"C":10.8,"D":100.0}},
    {"question": 2, "answer": null, "status": "MULTIPLE",
     "marked": ["A","B"], "fills": {"A":100.0,"B":100.0,"C":12.0,"D":16.3}}
  ],
  "summary": {"ok": 15, "blank": 0, "multiple": 1, "review": 0}
}
```

O campo **`fills`** (0–100) mostra o quão preenchida cada bolha ficou — é a sua
base de conferência: dá para ver exatamente por que cada questão foi classificada.

## Calibração / ajustes

Tudo o que depende do layout da folha está em **`omr/config.py`**:

- `COL_X`, `ROW_Y` — centros das bolhas no box canônico;
- `MARK_THRESHOLD` — a partir de quanto uma bolha conta como marcada (padrão 60 %);
- `REVIEW_LOW` — piso da zona ambígua (→ status `REVIEW`);
- `QUESTIONS_PER_BOX`, `NUM_BOXES` — geometria da folha.

Se a impressão da folha mudar, rode com `--debug` e ajuste `COL_X`/`ROW_Y`.

## Estrutura

```
omr-backend/
├── omr/
│   ├── config.py        # template + limiares (calibração)
│   └── engine.py        # pipeline OpenCV (detecção → warp → leitura)
├── app.py               # API FastAPI
├── test_cli.py          # teste por arquivo
├── make_test_images.py  # gera fotos sintéticas (branco/múltipla) p/ validar
├── samples/             # folha de referência
├── test_images/         # imagens de teste geradas
├── debug/               # saídas anotadas
└── requirements.txt
```

## Limitações / próximos passos

- A folha **não tem marcadores de canto** (fiducial markers). A detecção usa as
  bordas dos boxes, o que funciona bem, mas fotos muito tortas/escuras podem
  falhar. Adicionar quadradinhos pretos nos 4 cantos deixaria a leitura ainda
  mais robusta.
- Marcações a lápis fraco podem cair na zona `REVIEW` — ajuste `MARK_THRESHOLD`
  se for usar lápis em vez de caneta.
- O template atual assume 16 questões (8 por coluna). Outros formatos: ajustar
  `omr/config.py`.
```
