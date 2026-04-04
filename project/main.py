"""
FastAPI inference server for Ocular Disease Recognition.

Run:
    uvicorn project.main:app --reload
    # or from inside the project/ directory:
    uvicorn main:app --reload
"""

import io
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TARGET_COLS = ["N", "D", "G", "C", "A", "H", "M", "O"]

LABEL_NAMES = {
    "N": "Normal",
    "D": "Diabetic Retinopathy",
    "G": "Glaucoma",
    "C": "Cataract",
    "A": "Age-related Macular Degeneration",
    "H": "Hypertension",
    "M": "Myopia",
    "O": "Other diseases",
}

# Resolve model path relative to this file (project/ → repo root)
MODEL_PATH = Path(__file__).parent.parent / "best_multimodal_model.pth"

IMAGE_SIZE = 128
THRESHOLD = 0.5

# Training-time StandardScaler statistics for age
AGE_MEAN = 50.0
AGE_STD = 20.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Model Architecture (mirrors multi-v1.ipynb exactly)
# ---------------------------------------------------------------------------


class LightweightCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 3 → 32
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 2: 32 → 64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 3: 64 → 128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 4: 128 → 128
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, x):
        x = self.features(x)
        return x.view(x.size(0), -1)


class MultimodalFundusModel(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()
        self.image_encoder = LightweightCNN()

        self.tabular_encoder = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, 32),
            nn.ReLU(),
        )

        fused_dim = 128 * 2 + 32  # left(128) + right(128) + tabular(32)

        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, num_classes),
            nn.Sigmoid(),
        )

    def forward(self, left_img, right_img, tabular_data):
        left_feat = self.image_encoder(left_img)
        right_feat = self.image_encoder(right_img)
        tab_feat = self.tabular_encoder(tabular_data)
        fused = torch.cat([left_feat, right_feat, tab_feat], dim=1)
        return self.classifier(fused)


# ---------------------------------------------------------------------------
# Global model state
# ---------------------------------------------------------------------------

_model: Optional[MultimodalFundusModel] = None
_models_loaded: bool = False


def load_model() -> None:
    global _model, _models_loaded
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    _model = MultimodalFundusModel(num_classes=len(TARGET_COLS))
    _model.load_state_dict(checkpoint["model_state_dict"])
    _model.to(device)
    _model.eval()
    _models_loaded = True
    epoch = checkpoint.get("epoch", "?")
    val_acc = checkpoint.get("val_acc")
    if isinstance(val_acc, float):
        print(f"Model loaded from epoch {epoch} (val_acc={val_acc:.4f})")
    else:
        print(f"Model loaded from epoch {epoch}")


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

_transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

_hflip = transforms.RandomHorizontalFlip(p=1.0)


def preprocess_image(pil_img: Image.Image) -> torch.Tensor:
    """Return a (1, 3, H, W) tensor ready for the model."""
    return _transform(pil_img.convert("RGB")).unsqueeze(0).to(device)


async def read_image(upload: UploadFile) -> Image.Image:
    data = await upload.read()
    return Image.open(io.BytesIO(data))


# ---------------------------------------------------------------------------
# Lifespan (startup / shutdown)
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Ocular Disease Recognition API",
    description="Multi-label fundus disease classification (N/D/G/C/A/H/M/O).",
    version="1.0.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health/")
def health():
    """Check that the server is running and the model is loaded."""
    return {"status": "ok", "models_loaded": _models_loaded}


@app.post("/predict/")
async def predict(
    file: List[UploadFile] = File(..., description="1 or 2 fundus image files"),
    age: Optional[float] = Form(None, description="Patient age (optional)"),
    gender: Optional[str] = Form(
        None, description="Patient gender: M/Male or F/Female (optional)"
    ),
):
    """
    Classify ocular diseases from fundus image(s).

    - **1 image**: used as the left eye; a horizontal mirror serves as the right eye.
    - **2 images**: first = left eye, second = right eye.

    Optional `age` defaults to the training-set mean (50). Optional `gender`
    defaults to Male (0) when omitted.
    """
    if not _models_loaded or _model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")

    if len(file) == 0 or len(file) > 2:
        raise HTTPException(status_code=400, detail="Provide 1 or 2 image files.")

    # --- Images ---
    left_pil = await read_image(file[0])
    if len(file) == 2:
        right_pil = await read_image(file[1])
    else:
        # Symmetrical mirror for the missing eye
        right_pil = _hflip(left_pil.convert("RGB"))

    left_tensor = preprocess_image(left_pil)
    right_tensor = preprocess_image(right_pil)

    # --- Tabular features ---
    age_val = float(age) if age is not None else AGE_MEAN
    age_normalized = (age_val - AGE_MEAN) / AGE_STD

    gender_map = {"m": 0, "male": 0, "f": 1, "female": 1}
    gender_encoded = gender_map.get(gender.strip().lower(), 0) if gender else 0

    tabular_tensor = torch.tensor(
        [[age_normalized, float(gender_encoded)]], dtype=torch.float32
    ).to(device)

    # --- Inference ---
    with torch.no_grad():
        output = _model(left_tensor, right_tensor, tabular_tensor)
        probs = output[0].tolist()

    # --- Build response ---
    raw_outputs = {col: round(prob, 4) for col, prob in zip(TARGET_COLS, probs)}

    positive = [col for col, prob in zip(TARGET_COLS, probs) if prob >= THRESHOLD]

    # Fall back to top-1 class if nothing clears the threshold
    if not positive:
        top_idx = int(torch.tensor(probs).argmax().item())
        positive = [TARGET_COLS[top_idx]]

    prediction = ",".join(positive)
    label = ", ".join(LABEL_NAMES[col] for col in positive)

    return JSONResponse(
        content={
            "prediction": prediction,
            "label": label,
            "raw_outputs": raw_outputs,
        }
    )
