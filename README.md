# Ocular Disease Recognition

Multi-label fundus image classifier that detects 8 ocular diseases using a multimodal model (fundus images + patient metadata).

**Supported labels:**

| Code | Disease |
|------|---------|
| N | Normal |
| D | Diabetic Retinopathy |
| G | Glaucoma |
| C | Cataract |
| A | Age-related Macular Degeneration |
| H | Hypertension |
| M | Myopia |
| O | Other diseases |

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Running the API

```bash
uvicorn api:app --reload
```
or
```bash
uvicorn main:app --app-dir project --reload --port 8002
```

The server starts at `http://localhost:8000`. Interactive docs are available at `http://localhost:8000/docs`.

---

## API Endpoints

### `GET /health/`

Check that the server is running and the model is loaded.

**Response:**
```json
{
  "status": "ok",
  "models_loaded": true
}
```

---

### `POST /predict/`

Classify ocular diseases from one or two fundus images.

**Form fields:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | image file(s) | Yes | 1 or 2 fundus images. If 1 is provided, a horizontal mirror is used for the missing eye. |
| `age` | float | No | Patient age |
| `gender` | string | No | `M` / `Male` or `F` / `Female` |

**Response:**
```json
{
  "prediction": "D,G",
  "label": "Diabetic Retinopathy, Glaucoma",
  "raw_outputs": {
    "N": 0.0312,
    "D": 0.8745,
    "G": 0.6123,
    "C": 0.0211,
    "A": 0.0432,
    "H": 0.1034,
    "M": 0.0891,
    "O": 0.0543
  }
}
```

---

## Example Requests

### curl — single image

```bash
curl -X POST http://localhost:8000/predict/ \
  -F "file=@sample/007-0004-000.jpg" \
  -F "age=45" \
  -F "gender=M"
```

### curl — two images (left + right eye)

```bash
curl -X POST http://localhost:8000/predict/ \
  -F "file=@sample/007-0004-000.jpg" \
  -F "file=@sample/007-0007-000.jpg" \
  -F "age=45" \
  -F "gender=F"
```

### Python (`requests`)

```python
import requests

url = "http://localhost:8000/predict/"

with open("sample/007-0004-000.jpg", "rb") as f:
    response = requests.post(
        url,
        files=[("file", f)],
        data={"age": 45, "gender": "M"},
    )

print(response.json())
```

### Python — two images

```python
import requests

url = "http://localhost:8000/predict/"

with open("sample/007-0004-000.jpg", "rb") as left, \
     open("sample/007-0007-000.jpg", "rb") as right:
    response = requests.post(
        url,
        files=[("file", left), ("file", right)],
        data={"age": 60, "gender": "F"},
    )

print(response.json())
```
