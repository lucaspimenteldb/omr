"""
API FastAPI para o leitor de OMR.

Subir:
    uvicorn app:app --reload --host 0.0.0.0 --port 8000

Endpoints:
    GET  /health                      -> status do serviço
    POST /omr            (multipart)  -> lê a foto e devolve as marcações (JSON)
    POST /omr/debug      (multipart)  -> devolve a imagem anotada (PNG) para conferência
    GET  /docs                        -> Swagger UI (teste pelo navegador)

Campo do arquivo no form-data: "file"
"""
import io

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
import cv2

from omr import engine, OMRError

app = FastAPI(
    title="OMR Backend",
    description="Leitor de gabaritos (OpenCV). Detecta em branco, múltipla e única marcação.",
    version="1.0.0",
)


@app.get("/health")
def health():
    return {"status": "ok", "service": "omr-backend"}


@app.post("/omr")
async def read_omr(file: UploadFile = File(...)):
    """
    Recebe uma foto do gabarito (jpg/png) e devolve as marcações em JSON.

    status por questão:
      OK        -> exatamente uma alternativa marcada (campo answer preenchido)
      BLANK     -> nenhuma alternativa marcada
      MULTIPLE  -> duas ou mais alternativas marcadas
      REVIEW    -> preenchimento ambíguo, requer conferência humana
    """
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Arquivo vazio.")
    try:
        result = engine.decode_and_process(data)
    except OMRError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return {"filename": file.filename, **result}


@app.post("/omr/debug")
async def read_omr_debug(file: UploadFile = File(...)):
    """Igual ao /omr, mas devolve a IMAGEM anotada (PNG) para você conferir visualmente."""
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Arquivo vazio.")
    try:
        _, dbg = engine.decode_and_process(data, draw_debug=True)
    except OMRError as e:
        raise HTTPException(status_code=422, detail=str(e))
    ok, buf = cv2.imencode(".png", dbg)
    if not ok:
        raise HTTPException(status_code=500, detail="Falha ao codificar imagem de debug.")
    return StreamingResponse(io.BytesIO(buf.tobytes()), media_type="image/png")
