"""ALIS: servidor del frontend y la API."""
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router

FRONTEND = Path(__file__).resolve().parents[2] / "frontend"

app = FastAPI(title="ALIS API")
app.include_router(router)

app.mount(
    "/assets",
    StaticFiles(directory=FRONTEND / "assets"),
    name="assets",
)

@app.get("/", include_in_schema=False)
async def home():
    return FileResponse(FRONTEND / "index.html")

