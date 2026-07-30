"""
API FastAPI do leitor de gabaritos — modelo Cartão-Resposta Veloz (Anos finais).

Subir:
    uvicorn app:app --reload --host 0.0.0.0 --port 8000

Dois fluxos, um por página da folha. Os dois recebem a FOLHA INTEIRA
fotografada (campo `file`, multipart/form-data):

    POST /omr/objetiva        página 1 -> nº do aluno + Linguagens + Matemática
    POST /omr/redacao         página 2 -> nº do aluno + quadro de correção
    POST /omr/<fluxo>/debug   o mesmo, devolvendo a imagem anotada (PNG)
    GET  /health              status do serviço
    GET  /docs                Swagger UI

Se a foto enviada for da outra página, o endpoint recusa com 422 e diz qual
endpoint usar — em vez de devolver números sem sentido.
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
        "modelo": "Cartão-Resposta Veloz — Anos finais",
        "fluxos": {
            "objetiva": {
                "rota": "/omr/objetiva",
                "linguagens": sum(b.grade.n_linhas for b in T.AREAS["linguagens"]),
                "matematica": sum(b.grade.n_linhas for b in T.AREAS["matematica"]),
            },
            "redacao": {"rota": "/omr/redacao", "campos": list(T.ORDEM_CORRECAO)},
        },
    }


# --------------------------------------------------------------------------- #
# Fluxo 1 — respostas objetivas (página 1)
# --------------------------------------------------------------------------- #
@app.post(
    "/omr/objetiva",
    tags=["objetiva"],
    summary="Lê a página 1: número do aluno + Linguagens (1..25) e Matemática (1..26)",
    responses={200: {"content": {"application/json": {"example": EXEMPLO_OBJETIVA}}}},
)
async def omr_objetiva(file: UploadFile = File(..., description="Foto da folha inteira")):
    """
    Envie a foto da FOLHA INTEIRA da página 1 (com os 4 marcadores dos cantos).

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
    summary="Lê a página 2: número do aluno + quadro de correção do professor",
    responses={200: {"content": {"application/json": {"example": EXEMPLO_REDACAO}}}},
)
async def omr_redacao(file: UploadFile = File(..., description="Foto da folha inteira")):
    """
    Envie a foto da FOLHA INTEIRA da página 2 (PRODUÇÃO DE TEXTO).

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
