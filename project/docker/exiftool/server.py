# docker/exiftool/server.py
import subprocess, json, tempfile, os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

class ExifRequest(BaseModel):
    file_path: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/extract")
def extract(req: ExifRequest):
    if not os.path.exists(req.file_path):
        raise HTTPException(404, "Plik nie istnieje")
    result = subprocess.run(
        ["exiftool", "-json", "-n", req.file_path],
        capture_output=True, text=True
    )
    data = json.loads(result.stdout or "[]")
    return data[0] if data else {}