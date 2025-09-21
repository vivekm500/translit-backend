# translit_service.py
from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import cv2
import numpy as np
import pytesseract
from PIL import Image
import io
import os
import uuid
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate
from gtts import gTTS

# If tesseract is not in PATH, uncomment & set correct path:
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

app = FastAPI()

# ✅ CORS setup: allow POST + OPTIONS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For demo; replace with your frontend domain in prod
    allow_credentials=True,
    allow_methods=["*"],  # includes POST, GET, OPTIONS
    allow_headers=["*"],
)

# Explicit preflight handler (avoids 405 on OPTIONS)
@app.options("/{rest_of_path:path}")
async def preflight_handler(request: Request, rest_of_path: str):
    return JSONResponse({"message": "CORS preflight OK"})


# mapping frontend -> indic-transliteration
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

# --- Helpers ---
def pil_from_bytes(bts: bytes) -> Image.Image:
    return Image.open(io.BytesIO(bts)).convert("RGB")

def cv_from_pil(pil: Image.Image):
    arr = np.array(pil)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

def crop_image_cv(img_cv: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
    h_img, w_img = img_cv.shape[:2]
    x = max(0, int(round(x)))
    y = max(0, int(round(y)))
    w = max(1, int(round(w)))
    h = max(1, int(round(h)))
    x2 = min(w_img, x + w)
    y2 = min(h_img, y + h)
    return img_cv[y:y2, x:x2]

def preprocess_for_ocr(img: np.ndarray) -> np.ndarray:
    try:
        h, w = img.shape[:2]
        max_dim = 1600
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 9, 75, 75)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        th = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        processed = cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)
        processed = deskew(processed)
        return processed
    except Exception as e:
        print("preprocess error:", e)
        return img

def deskew(image: np.ndarray) -> np.ndarray:
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        coords = np.column_stack(np.where(gray > 0))
        if coords.shape[0] < 10:
            return image
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        (h, w) = image.shape[:2]
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h),
                                 flags=cv2.INTER_CUBIC,
                                 borderMode=cv2.BORDER_REPLICATE)
        return rotated
    except Exception:
        return image

def detect_script_from_text(text: str) -> str:
    if not text:
        return sanscript.DEVANAGARI
    for ch in text:
        if ch.strip():
            cp = ord(ch)
            if 0x0900 <= cp <= 0x097F: return sanscript.DEVANAGARI
            if 0x0B80 <= cp <= 0x0BFF: return sanscript.TAMIL
            if 0x0C00 <= cp <= 0x0C7F: return sanscript.TELUGU
            if 0x0C80 <= cp <= 0x0CFF: return sanscript.KANNADA
            if 0x0D00 <= cp <= 0x0D7F: return sanscript.MALAYALAM
            if 0x0A80 <= cp <= 0x0AFF: return sanscript.GUJARATI
            if 0x0980 <= cp <= 0x09FF: return sanscript.BENGALI
            if 0x0B00 <= cp <= 0x0B7F: return sanscript.ORIYA
            if 0x0A00 <= cp <= 0x0A7F: return sanscript.GURMUKHI
            break
    return sanscript.DEVANAGARI


# --- Endpoints ---

@app.post("/process-image")
async def process_image(
    image: UploadFile = File(...),
    x: float = Form(None),
    y: float = Form(None),
    width: float = Form(None),
    height: float = Form(None),
    target: str = Form("english"),
):
    try:
        contents = await image.read()
        pil = pil_from_bytes(contents)
        img_cv = cv_from_pil(pil)

        if x is not None and y is not None and width is not None and height is not None:
            img_cv = crop_image_cv(img_cv, x, y, width, height)

        processed = preprocess_for_ocr(img_cv)

        config = r'--oem 3 --psm 6'
        langs = "eng+hin+tel+tam+kan+mal+ben+guj+pan+ori"
        text = pytesseract.image_to_string(processed, lang=langs, config=config).strip()

        if not text:
            return JSONResponse({"error": "No text detected"}, status_code=200)

        detected = detect_script_from_text(text)
        if target not in target_map:
            return JSONResponse({"error": f"Unsupported target: {target}"}, status_code=400)

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


@app.post("/transliterate-text")
async def transliterate_text(body: dict):
    try:
        text = body.get("text", "").strip()
        target = body.get("target", "english")
        if not text:
            return JSONResponse({"error": "No text provided"}, status_code=400)
        if target not in target_map:
            return JSONResponse({"error": f"Unsupported target: {target}"}, status_code=400)

        detected = detect_script_from_text(text)
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


@app.post("/speak")
async def speak(body: dict):
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
