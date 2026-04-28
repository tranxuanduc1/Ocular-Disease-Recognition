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

import timm
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

MODEL_PATH = (
    Path(__file__).parent.parent
    / "model_v2_result/inceptionresnet_v2/checkpoints/inceptionresnet_best.pth"
)

IMAGE_SIZE = 299

# Optimal thresholds per class (tuned on validation set, from 07_evaluation.ipynb)
THRESHOLDS = {
    "N": 0.35, "D": 0.45, "G": 0.70, "C": 0.85,
    "A": 0.90, "H": 0.70, "M": 0.65, "O": 0.45,
}

# From model_v2_result/inceptionresnet_v2/norm_params.json
AGE_MEAN = 57.821443130860175
AGE_STD = 11.769395711092763

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Model Architecture
# ---------------------------------------------------------------------------


class SiameseMultimodalNet(nn.Module):
    def __init__(
        self,
        backbone_name,
        proj_dim=128,
        tabular_dim=16,
        dropout=0.4,
        num_classes=8,
        use_batchnorm=False,
    ):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name, pretrained=False, num_classes=0, global_pool="avg"
        )
        feat_dim = self.backbone.num_features
        self.projector = nn.Sequential(
            nn.Linear(feat_dim, proj_dim), nn.ReLU(inplace=True), nn.Dropout(dropout)
        )
        self.tabular_encoder = nn.Sequential(
            nn.Linear(3, tabular_dim), nn.ReLU(inplace=True)
        )
        fused_dim = 2 * proj_dim + tabular_dim
        if use_batchnorm:
            self.classifier = nn.Sequential(
                nn.Linear(fused_dim, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(64, num_classes),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(fused_dim, 64),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(64, num_classes),
            )

    def forward_one(self, x):
        return self.projector(self.backbone(x))

    def forward(self, left, right, tabular):
        img_feat = torch.cat([self.forward_one(left), self.forward_one(right)], dim=1)
        return self.classifier(torch.cat([img_feat, self.tabular_encoder(tabular)], dim=1))


# ---------------------------------------------------------------------------
# Global model state
# ---------------------------------------------------------------------------

_model: Optional[SiameseMultimodalNet] = None
_models_loaded: bool = False


def load_model() -> None:
    global _model, _models_loaded
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    _model = SiameseMultimodalNet(
        backbone_name="inception_resnet_v2",
        proj_dim=128,
        tabular_dim=16,
        dropout=0.4,
        num_classes=len(TARGET_COLS),
        use_batchnorm=True,
    )
    _model.load_state_dict(checkpoint["model_state_dict"])
    _model.to(device)
    _model.eval()
    _models_loaded = True
    epoch = checkpoint.get("epoch", "?")
    val_auc = checkpoint.get("val_auc")
    if isinstance(val_auc, float):
        print(f"Model loaded from epoch {epoch} (val_auc={val_auc:.4f})")
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


def encode_gender(gender_str: Optional[str]) -> List[float]:
    """One-hot encode gender: [1, 0] = Male, [0, 1] = Female."""
    s = gender_str.strip().lower() if gender_str else ""
    return [1.0, 0.0] if s in ("male", "m") else [0.0, 1.0]


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
    version="2.0.0",
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

    Optional `age` defaults to the training-set mean (~58). Optional `gender`
    defaults to Female when omitted.
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
        right_pil = _hflip(left_pil.convert("RGB"))

    left_tensor = preprocess_image(left_pil)
    right_tensor = preprocess_image(right_pil)

    # --- Tabular features ---
    age_val = float(age) if age is not None else AGE_MEAN
    age_normalized = (age_val - AGE_MEAN) / AGE_STD

    gender_encoded = encode_gender(gender)
    tabular_tensor = torch.tensor(
        [[age_normalized] + gender_encoded], dtype=torch.float32
    ).to(device)

    # --- Inference ---
    with torch.no_grad():
        logits = _model(left_tensor, right_tensor, tabular_tensor)
        probs = torch.sigmoid(logits)[0].tolist()

    # --- Build response ---
    raw_outputs = {col: round(prob, 4) for col, prob in zip(TARGET_COLS, probs)}

    positive = [col for col, prob in zip(TARGET_COLS, probs) if prob >= THRESHOLDS[col]]

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
