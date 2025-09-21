# translit_service.py
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate
from gtts import gTTS
import os
import uuid

app = FastAPI()

# ✅ Allow CORS (replace * with your frontend URL for security)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # e.g. ["https://translit-frontend.vercel.app"]
    allow_credentials=True,
    allow_methods=["*"],  # allows POST, OPTIONS, etc.
    allow_headers=["*"],
)

# ---------- Config ----------

# Mapping frontend -> sanscript scheme
target_map = {
    "hindi": sanscript.DEVANAGARI,
    "english": sanscript.ITRANS,
    "telugu": sanscript.TELUGU,
    "tamil": sanscript.TAMIL,
    "kannada": sanscript.KANNADA,
    "malayalam": sanscript.MALAYALAM,
    "gujarati": sanscript.GUJARATI,
    "bengali": sanscript.BENGALI,
    "oriya": sanscript.ORIYA,
    "gurmukhi": sanscript.GURMUKHI,
    "iast": sanscript.IAST,
}

# gTTS language codes
gtts_langs = {
    "hindi": "hi",
    "english": "en",
    "telugu": "te",
    "tamil": "ta",
    "kannada": "kn",
    "malayalam": "ml",
    "gujarati": "gu",
    "bengali": "bn",
    "oriya": "or",
    "gurmukhi": "pa",
    "iast": "en",
}


# ---------- Helpers ----------

def detect_script(text: str) -> str:
    """Naive Unicode block detection"""
    for ch in text.strip():
        cp = ord(ch)
        if 0x0900 <= cp <= 0x097F:
            return sanscript.DEVANAGARI
        if 0x0B80 <= cp <= 0x0BFF:
            return sanscript.TAMIL
        if 0x0C00 <= cp <= 0x0C7F:
            return sanscript.TELUGU
        if 0x0C80 <= cp <= 0x0CFF:
            return sanscript.KANNADA
        if 0x0D00 <= cp <= 0x0D7F:
            return sanscript.MALAYALAM
        if 0x0A80 <= cp <= 0x0AFF:
            return sanscript.GUJARATI
        if 0x0980 <= cp <= 0x09FF:
            return sanscript.BENGALI
        if 0x0B00 <= cp <= 0x0B7F:
            return sanscript.ORIYA
        if 0x0A00 <= cp <= 0x0A7F:
            return sanscript.GURMUKHI
    return sanscript.DEVANAGARI


# ---------- API Endpoints ----------

@app.post("/transliterate-text")
async def transliterate_text(body: dict):
    """Transliterate plain text"""
    try:
        text = body.get("text", "").strip()
        target = body.get("target", "english")

        if not text:
            return JSONResponse({"error": "No text provided"}, status_code=400)
        if target not in target_map:
            return JSONResponse({"error": f"Unsupported target: {target}"}, status_code=400)

        detected = detect_script(text)
        transliterated_user = transliterate(text, detected, target_map[target])
        transliterated_english = transliterate(text, detected, sanscript.IAST)

        return {
            "input": text,
            "detected_script": detected,
            "transliterated_user": transliterated_user,
            "transliterated_english": transliterated_english,
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/process-image")
async def process_image(image: UploadFile = File(...), target: str = Form(...)):
    """Dummy OCR endpoint (replace with Tesseract/EasyOCR later).
    For now, just returns 'OCR not implemented'."""
    return {"error": "OCR not implemented in this demo", "target": target}


@app.post("/speak")
async def speak(body: dict):
    """Text-to-speech with gTTS"""
    try:
        text = body.get("text", "").strip()
        target = body.get("target", "english")

        if not text:
            return JSONResponse({"error": "No text provided"}, status_code=400)
        if target not in gtts_langs:
            return JSONResponse({"error": f"No TTS available for {target}"}, status_code=400)

        lang_code = gtts_langs[target]
        tts = gTTS(text=text, lang=lang_code)

        filename = f"speech_{uuid.uuid4().hex}.mp3"
        out_dir = "tmp"
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, filename)
        tts.save(out_path)

        return FileResponse(out_path, media_type="audio/mpeg", filename="speech.mp3")
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ---------- Run (for local) ----------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
