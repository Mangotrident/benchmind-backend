from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from io import StringIO
from app.core import suggest_next

app = FastAPI(title="BenchMind API", version="1.0")

# --- CORS so your Netlify frontend can call the API ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for first deploy; tighten later to your Netlify domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    content = await file.read()
    df = pd.read_csv(StringIO(content.decode("utf-8")))
    return {"rows": len(df), "columns": list(df.columns)}

@app.post("/suggest_next")
async def suggest_next_api(file: UploadFile = File(...), k: int = 5):
    content = await file.read()
    df = pd.read_csv(StringIO(content.decode("utf-8")))
    top = suggest_next(df, bounds=(1e-3, 100), k=k)
    return top.to_dict(orient="records")
