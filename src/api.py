from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import PyPDF2
import docx
import io
import re

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(title="AI Text Detector")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="src/templates")

# ── Load RoBERTa model ────────────────────────────────────────────────────────
from huggingface_hub import snapshot_download
import os

DEVICE    = "cpu"  # Render free tier has no GPU
MODEL_DIR = "roberta_model"

if not os.path.exists(MODEL_DIR):
    print("Downloading model from HuggingFace...")
    snapshot_download(
        repo_id="gazalyadav/ai-text-detector-roberta",
        local_dir=MODEL_DIR
    )

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model     = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(DEVICE)
model.eval()
print("Model ready!")

# ── Text cleaning ─────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ── Inference ─────────────────────────────────────────────────────────────────
def predict(text: str) -> dict:
    cleaned = clean_text(text)
    if len(cleaned.split()) < 10:
        return {"error": "Text too short. Please provide at least 10 words."}

    inputs = tokenizer(
        cleaned,
        return_tensors="pt",
        max_length=256,
        truncation=True,
        padding="max_length"
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits
        probs  = torch.softmax(logits, dim=1)[0]

    ai_score    = round(probs[1].item() * 100, 2)
    human_score = round(probs[0].item() * 100, 2)
    label       = "AI-Generated" if ai_score > 50 else "Human-Written"

    return {
        "label"      : label,
        "ai_score"   : ai_score,
        "human_score": human_score,
        "confidence" : max(ai_score, human_score),
    }

# ── File text extractors ──────────────────────────────────────────────────────
def extract_from_pdf(file_bytes: bytes) -> str:
    reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
    return " ".join(page.extract_text() or "" for page in reader.pages)

def extract_from_docx(file_bytes: bytes) -> str:
    doc = docx.Document(io.BytesIO(file_bytes))
    return " ".join(p.text for p in doc.paragraphs)

def extract_from_txt(file_bytes: bytes) -> str:
    return file_bytes.decode("utf-8", errors="ignore")

# ── Routes ────────────────────────────────────────────────────────────────────
class TextInput(BaseModel):
    text: str

@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html"
    )

@app.post("/detect/text")
def detect_text(inp: TextInput):
    return predict(inp.text)

@app.post("/detect/file")
async def detect_file(file: UploadFile = File(...)):
    contents = await file.read()
    filename = file.filename.lower()

    if filename.endswith(".pdf"):
        text = extract_from_pdf(contents)
    elif filename.endswith(".docx"):
        text = extract_from_docx(contents)
    elif filename.endswith(".txt"):
        text = extract_from_txt(contents)
    else:
        return {"error": "Unsupported file type. Use .pdf, .docx, or .txt"}

    if not text.strip():
        return {"error": "Could not extract text from file."}

    result = predict(text)
    result["filename"] = file.filename
    result["word_count"] = len(text.split())
    return result