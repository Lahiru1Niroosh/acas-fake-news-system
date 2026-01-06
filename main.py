from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
import cv2
from io import BytesIO
from PIL import Image
from scipy import fftpack
from skimage import color
from skimage.feature import local_binary_pattern
import os
import math
from dotenv import load_dotenv

load_dotenv() 

app = FastAPI(title="AI-vs-Real Image Detector with Explanations")

# --- CONFIG ---
# Default path to your saved model (replace with actual .keras or .h5)
MODEL_PATH = os.getenv("MODEL_PATH", "Model/cnn_image_classifier.keras")

# Developer note: include uploaded file path from conversation history as a file URL
# (the system will transform this path to an accessible URL if needed).
UPLOADED_FILE_URL = os.getenv("FILE_PATH", "Model/REAL_VS_FAKE(AI)_Image_Classifier.ipynb")

# Model will be lazily loaded
_model = None
INPUT_SIZE = (256, 256)  # default; if your model uses different size, set MODEL_INPUT_SIZE env var


class PredictionResponse(BaseModel):
    result: str
    confidence: float
    reasons: list
    detailed: dict


def load_model(path: str):
    global _model
    if _model is None:
        # Try load; raise helpful errors if fails
        try:
            _model = tf.keras.models.load_model(path)
        except Exception as e:
            raise RuntimeError(f"Failed to load model at {path}: {e}")

        # Force-build model using a dummy input (use correct input size if available)
        # try to infer input size from model, else use INPUT_SIZE
        try:
            # If model.input is None, call with dummy
            if not hasattr(_model, 'input') or _model.input is None:
                dummy_shape = (1, INPUT_SIZE[0], INPUT_SIZE[1], 3)
                _model(np.zeros(dummy_shape, dtype=np.float32))
        except Exception:
            # last resort: try predict with zeros
            try:
                _model.predict(np.zeros((1, INPUT_SIZE[0], INPUT_SIZE[1], 3), dtype=np.float32))
            except Exception:
                pass

    return _model


# ---------- Image utility functions ----------

def read_imagefile(file_bytes) -> np.ndarray:
    img = Image.open(BytesIO(file_bytes)).convert('RGB')
    return np.array(img)


def preprocess_for_model(img: np.ndarray, target_size=INPUT_SIZE):
    # Resize and scale to [0,1]
    img_resized = cv2.resize(img, (target_size[1], target_size[0]))
    img_norm = img_resized.astype('float32') / 255.0
    return np.expand_dims(img_norm, axis=0), img_resized


# ---------- Heuristics for explanations ----------

def texture_score(img: np.ndarray) -> float:
    """Compute a texture irregularity score using Local Binary Patterns (LBP).
    Returns a score in [0,1] where higher indicates more "synthetic-like" uniform textures.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # LBP
    lbp = local_binary_pattern(gray, P=8, R=1, method='ror')
    # Histogram of LBP
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 2**8 + 1), density=True)
    # Entropy of LBP histogram (lower entropy often in synthetic uniform textures)
    hist_nonzero = hist[hist > 0]
    if hist_nonzero.size == 0:
        return 0.0
    entropy = -np.sum(hist_nonzero * np.log(hist_nonzero))
    # Normalize entropy to [0,1] via log(max_bins)
    max_entropy = math.log(len(hist))
    score = 1.0 - (entropy / max_entropy)
    return float(np.clip(score, 0.0, 1.0))


def edge_anomaly_score(img: np.ndarray) -> float:
    """Compute edge density and variance. Synthetic images often have edge inconsistencies.
    Returns [0,1] where higher = more anomalous.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edge_density = edges.mean()  # between 0 and 1
    # Laplacian variance (blurriness measure)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    # Normalize lap_var using heuristic upper bound
    lap_norm = np.tanh(lap_var / 100.0)
    # Combine
    score = float(np.clip(edge_density * 0.6 + (1 - lap_norm) * 0.4, 0.0, 1.0))
    return score


def color_anomaly_score(img: np.ndarray) -> float:
    """Compute color histogram uniformity & saturation anomalies. Returns [0,1].
    Synthetic images sometimes have over-smooth color distributions.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    saturation = hsv[:,:,1].astype('float32') / 255.0
    sat_mean = saturation.mean()
    sat_std = saturation.std()
    # If saturation std is low while mean is moderate-high -> suspicious
    score = float(np.clip((1 - sat_std) * sat_mean, 0.0, 1.0))
    return score


def fft_anomaly_score(img: np.ndarray) -> float:
    """Analyze frequency spectrum: synthetic images may have unusual high-frequency spikes.
    Returns [0,1].
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype('float32')
    f = fftpack.fft2(gray)
    fshift = fftpack.fftshift(f)
    magnitude_spectrum = np.abs(fshift)
    # Low vs high frequency energy
    h, w = magnitude_spectrum.shape
    cy, cx = h//2, w//2
    r = min(cy, cx)
    # radial mask: low-frequency circle radius = r*0.2
    low_r = int(r * 0.2)
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
    low_mask = dist <= low_r
    high_mask = dist >= (r * 0.6)
    low_energy = magnitude_spectrum[low_mask].sum() + 1e-8
    high_energy = magnitude_spectrum[high_mask].sum() + 1e-8
    ratio = high_energy / low_energy
    # Normalize ratio with tanh
    score = float(np.tanh((ratio - 0.5) / 5.0) * 0.5 + 0.5)
    return float(np.clip(score, 0.0, 1.0))


def combined_heuristic(img: np.ndarray) -> dict:
    t = texture_score(img)
    e = edge_anomaly_score(img)
    c = color_anomaly_score(img)
    f = fft_anomaly_score(img)
    # Combine into a single synthetic-likelihood heuristic
    # Weighted sum (weights chosen heuristically)
    combined = 0.35 * t + 0.25 * e + 0.2 * c + 0.2 * f
    return {"texture_score": t, "edge_score": e, "color_score": c, "fft_score": f, "combined_score": float(np.clip(combined,0,1))}


# ---------- API endpoints ----------

@app.get('/')
def read_root():
    return {"message": "Welcome to the AI-vs-Real Image Detector API."}

@app.get("/health")
async def health():
    return {"status": "ok", "model_path_example": UPLOADED_FILE_URL}


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    # Read bytes
    try:
        contents = await file.read()
        img = read_imagefile(contents)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image upload: {e}")

    # Preprocess
    img_for_model, img_resized = preprocess_for_model(img)

    # Load model
    try:
        model = load_model(MODEL_PATH)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Predict
    try:
        preds = model.predict(img_for_model)
        # Handle binary or softmax
        if preds.shape[-1] == 1:
            confidence = float(preds[0,0])  # probability of real (label=1)
            label = 'real' if confidence >= 0.5 else 'fake'
            # Adjust confidence to always reflect predicted class
            confidence = confidence if label == 'real' else 1.0 - confidence
        else:
            # train_ds.class_names = ['fake', 'real'] → 0=fake, 1=real
            CLASS_MAP = {0: 'fake', 1: 'real'}
            idx = int(np.argmax(preds[0]))
            label = CLASS_MAP[idx]
            confidence = float(np.max(preds[0]))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model prediction failed: {e}")

    # Heuristic analysis
    heur = combined_heuristic(img_resized)

    # Build textual reasons
    reasons = []
    # Simple textual reasons (A)
    if heur['combined_score'] > 0.6:
        reasons.append("Multiple artifact signals detected (texture, edges, color, or frequency) — likely AI-generated.")
    else:
        reasons.append("No strong artifact signals detected; image characteristics align more with natural photos.")

    # Add targeted reasons based on thresholds (B)
    if heur['texture_score'] > 0.45:
        reasons.append("Unnatural/repetitive texture patterns detected (LBP entropy low).")
    if heur['edge_score'] > 0.5:
        reasons.append("Edge density/blur inconsistencies detected between object boundaries and background.")
    if heur['color_score'] > 0.4:
        reasons.append("Color distribution anomalies found: saturation uniformity uncommon in real photos.")
    if heur['fft_score'] > 0.5:
        reasons.append("High-frequency spectral peaks found — common in generative models (GAN artifacts).")

    # If model predicted fake but heuristics are low, mention model-based reason
    if label == 'fake' and heur['combined_score'] <= 0.4:
        reasons.append("Model's learned features indicate synthetic patterns even though heuristic tests are weak — this can happen if model learned subtle cues.")

    response = {
        "result": label,
        "confidence": round(confidence, 4),
        "reasons": reasons,
        "detailed": heur
    }

    return JSONResponse(status_code=200, content=response)


# Quick CLI test helper (not an endpoint)
if __name__ == '__main__':
    print("FastAPI app file. Run with: uvicorn fastapi_ai_detector:app --reload")
    print(f"Example uploaded-file-path (from conversation): {UPLOADED_FILE_URL}")
