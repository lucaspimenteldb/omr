"""
API FastAPI do leitor de gabaritos — Cartão-Resposta Veloz.

Subir:
    uvicorn app:app --reload --host 0.0.0.0 --port 8000

Dois fluxos, um por página da folha. Os dois recebem a FOLHA INTEIRA
fotografada (campo `file`, multipart/form-data):

    POST /omr/objetiva        respostas -> nº do aluno + Linguagens + Matemática
    POST /omr/redacao         redação   -> nº do aluno + quadro de correção
    POST /omr/<fluxo>/debug   o mesmo, devolvendo a imagem anotada (PNG)
    GET  /health              status do serviço e os modelos suportados
    GET  /docs                Swagger UI

Dois MODELOS de folha são atendidos e reconhecidos automaticamente:

    anos_iniciais   Linguagens 1..21, Matemática 1..22   (não tem redação)
    anos_finais     Linguagens 1..25, Matemática 1..26   (+ redação)

O modelo detectado volta no campo `model` da resposta. Se a foto for da outra
página — ou não for reconhecida como nenhum dos modelos — o endpoint recusa com
422 e diz o que fazer, em vez de devolver números sem sentido.
"""
import io

import cv2
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from omr import OMRError, decodificar, ler_fluxo
from omr import template as T

app = FastAPI(
    title="OMR Backend — Cartão-Resposta Veloz",
    description=__doc__,
    version="2.0.0",
)

EXEMPLO_OBJETIVA = {
    "filename": "folha.jpg",
    "flow": "objetiva",
    "model": "anos_finais",
    "model_title": "Cartão-Resposta Veloz — Anos Finais",
    "student_number": {
        "value": "1207",
        "raw": "______1207",
        "status": "OK",
        "digits": [{"position": 7, "digit": "1", "status": "OK", "marked": ["1"],
                    "fills": {"0": 11.2, "1": 99.8}}],
    },
    "sections": [
        {"name": "linguagens", "num_questions": 25,
         "summary": {"ok": 24, "blank": 1, "multiple": 0, "review": 0},
         "results": [{"question": 1, "answer": "D", "status": "OK", "marked": ["D"],
                      "fills": {"A": 12.1, "B": 10.4, "C": 11.7, "D": 99.6}}]},
        {"name": "matematica", "num_questions": 26, "summary": {}, "results": []},
    ],
    "summary": {"ok": 49, "blank": 1, "multiple": 1, "review": 0},
    "alignment": {"fiducials": "fiduciais", "coverage": 0.98, "rotation": 0},
}

EXEMPLO_REDACAO = {
    "filename": "redacao.jpg",
    "flow": "redacao",
    "model": "anos_finais",
    "student_number": {"value": "1207", "raw": "______1207", "status": "OK", "digits": []},
    "correction": {
        "situacao": {"value": "C", "status": "OK", "marked": ["C"],
                     "fills": {"A": 9.9, "B": 10.2, "C": 99.1, "D": 8.8, "E": 9.4}},
        "competencia_01": {"value": "3", "status": "OK", "marked": ["3"],
                           "fills": {"0": 9.1, "1": 8.7, "2": 9.9, "3": 98.7, "4": 9.2},
                           "level": 3},
    },
    "summary": {"ok": 5, "blank": 1, "multiple": 0, "review": 0},
    "alignment": {"fiducials": "fiduciais", "coverage": 0.97, "rotation": 0},
}


async def _processar(fluxo: str, file: UploadFile, debug: bool):
    data = await file.read()
    try:
        imagem = decodificar(data)
        return ler_fluxo(fluxo, imagem, debug=debug)
    except OMRError as e:
        raise HTTPException(status_code=422, detail=str(e)) from None


def _png(imagem) -> StreamingResponse:
    ok, buf = cv2.imencode(".png", imagem)
    if not ok:
        raise HTTPException(status_code=500, detail="Falha ao codificar a imagem de debug.")
    return StreamingResponse(io.BytesIO(buf.tobytes()), media_type="image/png")


@app.get("/health", tags=["serviço"])
def health():
    return {
        "status": "ok",
        "service": "omr-backend",
        
        "modelos": {
            m.nome: {
                "titulo": m.titulo,
                "fluxos": {
                    fluxo: (
                        {area: sum(b.grade.n_linhas for b in blocos)
                         for area, blocos in folha.areas.items()}
                        if folha.areas else {"campos": list(T.ORDEM_CORRECAO)}
                    )
                    for fluxo, folha in m.folhas.items()
                },
            }
            for m in T.MODELOS.values()
        },
        "rotas": {"objetiva": "/omr/objetiva", "redacao": "/omr/redacao"},
    }


# --------------------------------------------------------------------------- #
# Fluxo 1 — respostas objetivas (página 1)
# --------------------------------------------------------------------------- #
@app.post(
    "/omr/objetiva",
    tags=["objetiva"],
    summary="Lê a página de respostas (qualquer modelo): nº do aluno + Linguagens e Matemática",
    responses={200: {"content": {"application/json": {"example": EXEMPLO_OBJETIVA}}}},
)
async def omr_objetiva(file: UploadFile = File(..., description="Foto da folha inteira")):
    """
    Envie a foto da FOLHA INTEIRA (com os 4 marcadores dos cantos). O modelo —
    Anos Iniciais (21 + 22 questões) ou Anos Finais (25 + 26) — é reconhecido
    pela geometria da folha e volta em `model`.

    Status possíveis por questão:

    | status | significado |
    |---|---|
    | `OK` | exatamente uma alternativa marcada (vem em `answer`) |
    | `BLANK` | nenhuma alternativa marcada |
    | `MULTIPLE` | duas ou mais marcadas |
    | `REVIEW` | preenchimento ambíguo — conferir a olho |

    `fills` (0–100) é o quanto cada bolha ficou preenchida: é a base de
    conferência de qualquer classificação.
    """
    resultado = await _processar("objetiva", file, debug=False)
    return {"filename": file.filename, **resultado}


@app.post("/omr/objetiva/debug", tags=["objetiva"],
          summary="Igual a /omr/objetiva, mas devolve a foto anotada (PNG)")
async def omr_objetiva_debug(file: UploadFile = File(...)):
    """Devolve a própria foto com cada bolha lida circulada — verde = marcada,
    cinza = vazia, laranja = ambígua, e a moldura magenta mostra onde o motor
    entendeu que estão os marcadores fiduciais."""
    _, imagem = await _processar("objetiva", file, debug=True)
    return _png(imagem)


# --------------------------------------------------------------------------- #
# Fluxo 2 — correção da redação (página 2)
# --------------------------------------------------------------------------- #
@app.post(
    "/omr/redacao",
    tags=["redação"],
    summary="Lê a página de produção de texto: nº do aluno + quadro de correção",
    responses={200: {"content": {"application/json": {"example": EXEMPLO_REDACAO}}}},
)
async def omr_redacao(file: UploadFile = File(..., description="Foto da folha inteira")):
    """
    Envie a foto da FOLHA INTEIRA da página de PRODUÇÃO DE TEXTO. Só o modelo
    Anos Finais tem essa página; folha de Anos Iniciais é recusada com 422.

    Devolve o número do aluno e os 6 campos do quadro *CORREÇÃO – uso exclusivo
    do professor avaliador*: `situacao` (A–E) e `competencia_01`..`competencia_05`
    (0–4, também em `level` como inteiro). As linhas escritas da redação não são
    lidas — só o quadro de correção.
    """
    resultado = await _processar("redacao", file, debug=False)
    return {"filename": file.filename, **resultado}


@app.post("/omr/redacao/debug", tags=["redação"],
          summary="Igual a /omr/redacao, mas devolve a foto anotada (PNG)")
async def omr_redacao_debug(file: UploadFile = File(...)):
    """Foto anotada com as bolhas lidas do quadro de correção e do número do aluno."""
    _, imagem = await _processar("redacao", file, debug=True)
    return _png(imagem)


# --------------------------------------------------------------------------- #
# Rota antiga
# --------------------------------------------------------------------------- #
@app.post("/omr", tags=["serviço"], include_in_schema=False)
async def omr_legado(file: UploadFile = File(None)):
    raise HTTPException(
        status_code=410,
        detail="A rota /omr era do modelo antigo de folha (16 questões). "
               "Use /omr/objetiva (respostas) ou /omr/redacao (correção da redação).",
    )
