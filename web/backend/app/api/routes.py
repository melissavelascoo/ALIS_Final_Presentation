"""Rutas HTTP para reconocimiento y estado de los modelos."""
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from app.core.config import MAX_FRAME_BYTES
from app.services.classifier import recognition

router = APIRouter(prefix="/api")


@router.get("/health")
async def health():
    return {"status": "ok", "service": "ALIS"}


@router.get("/model-status")
async def model_status():
    return await run_in_threadpool(recognition.status)


@router.post("/reset")
async def reset():
    return await run_in_threadpool(recognition.reset)


@router.post("/predict")
async def predict(
    frame: UploadFile = File(...),
    mode: str = Form("both"),
):
    normalized_mode = mode.strip().lower()
    if normalized_mode not in {"auto", "both", "static", "dynamic"}:
        raise HTTPException(status_code=400, detail="Modo no válido.")
    if frame.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(status_code=415, detail="Se requiere una imagen JPEG, PNG o WEBP.")
    content = await frame.read(MAX_FRAME_BYTES + 1)
    if not content:
        raise HTTPException(status_code=400, detail="La imagen está vacía.")
    if len(content) > MAX_FRAME_BYTES:
        raise HTTPException(status_code=413, detail="La imagen es demasiado grande.")
    try:
        return await run_in_threadpool(recognition.predict, content, normalized_mode)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc