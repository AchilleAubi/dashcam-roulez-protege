# server.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import subprocess
import os
import datetime

app = FastAPI()

origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Dossier de stockage ----
UPLOAD_DIR = Path("recordings")
UPLOAD_DIR.mkdir(exist_ok=True)

app.mount("/files", StaticFiles(directory=str(UPLOAD_DIR)), name="files")


def file_info(p: Path):
    stat = p.stat()
    return {
        "name": p.name,
        "size": stat.st_size,
        "createdAt": datetime.datetime.fromtimestamp(stat.st_ctime).isoformat(),
        "url": f"/files/{p.name}",
    }


@app.get("/api/recordings")
def list_recordings():
    files = sorted(
        [p for p in UPLOAD_DIR.iterdir() if p.is_file() and p.suffix.lower() in {".mp4", ".webm"}],
        key=lambda p: p.stat().st_ctime,
        reverse=True
    )
    return [file_info(p) for p in files]


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    raw_path = UPLOAD_DIR / file.filename
    with raw_path.open("wb") as f:
        f.write(await file.read())

    mp4_path = raw_path.with_suffix(".mp4")
    cmd = [
        "ffmpeg", "-y", "-i", str(raw_path),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-c:a", "aac",
        str(mp4_path)
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        try:
            raw_path.unlink(missing_ok=True)
        except Exception:
            pass

        return JSONResponse({"ok": True, "name": mp4_path.name, "url": f"/files/{mp4_path.name}"})
    except subprocess.CalledProcessError as e:
        return JSONResponse({"ok": False, "error": e.stderr.decode("utf-8")}, status_code=500)


@app.delete("/api/recordings/{name}")
def delete_recording(name: str):
    p = UPLOAD_DIR / name
    if not p.exists():
        raise HTTPException(status_code=404, detail="Fichier introuvable")
    p.unlink()
    return {"ok": True}
