from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
import requests
import subprocess
import time
import socket
import shlex
from urllib.parse import urlparse
import joblib
import pandas as pd
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import uuid

app = FastAPI()

# load .env if present
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

# LLM config from env
LLAMA_SERVER_URL = os.getenv('LLAMA_SERVER_URL')
LLAMA_API_PATH = os.getenv('LLAMA_API_PATH', '/v1/generate')
LLAMA_SERVER_EXE = os.getenv('LLAMA_SERVER_EXE')
LLAMA_MODEL_PATH = os.getenv('LLAMA_MODEL_PATH')
LLAMA_SERVER_ARGS = os.getenv('LLAMA_SERVER_ARGS', '')

# process handle if we spawn the server
LLAMA_PROCESS = None


def is_port_open(host: str, port: int, timeout: float = 0.8) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:
        return False


# serve static files
static_dir = os.path.join(os.path.dirname(__file__), 'static')
if not os.path.exists(static_dir):
    os.makedirs(static_dir)
app.mount('/static', StaticFiles(directory=static_dir), name='static')

# We'll serve the frontend as a static HTML file and provide a /config endpoint

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'xgboost_pipeline.pkl')
PIPELINE = joblib.load(MODEL_PATH)
SCALER = PIPELINE.named_steps.get('scaler')
MODEL = PIPELINE.named_steps.get('model')

FEATURES = [
    "gender",
    "age",
    "hypertension",
    "heart_disease",
    "ever_married",
    "work_type",
    "Residence_type",
    "avg_glucose_level",
    "bmi",
    "smoking_status",
]

# Provide metadata for frontend: examples and options for categorical fields.
# NOTE: these numeric option values match the encoded integers used in the training pipeline
FEATURE_CONFIG = {
    "gender": {
        "type": "categorical",
        "options": [
            {"label": "Female", "value": 0},
            {"label": "Male", "value": 1},
            {"label": "Other", "value": 2}
        ],
        "example": 1
    },
    "ever_married": {
        "type": "categorical",
        "options": [
            {"label": "No", "value": 0},
            {"label": "Yes", "value": 1}
        ],
        "example": 1
    },
    "work_type": {
        "type": "categorical",
        "options": [
            {"label": "Government job", "value": 0},
            {"label": "Never worked", "value": 1},
            {"label": "Private", "value": 2},
            {"label": "Self-employed", "value": 3},
            {"label": "Children", "value": 4}
        ],
        "example": 2
    },
    "Residence_type": {
        "type": "categorical",
        "options": [
            {"label": "Rural", "value": 0},
            {"label": "Urban", "value": 1}
        ],
        "example": 1
    },
    "smoking_status": {
        "type": "categorical",
        "options": [
            {"label": "never smoked", "value": 0},
            {"label": "formerly smoked", "value": 1},
            {"label": "smokes", "value": 2},
            {"label": "Unknown", "value": 3}
        ],
        "example": 1
    },
    # Numeric fields with example values
    "age": {"type": "numeric", "example": 67},
    "hypertension": {"type": "numeric", "example": 0},
    "heart_disease": {"type": "numeric", "example": 1},
    "avg_glucose_level": {"type": "numeric", "example": 228.69},
    "bmi": {"type": "numeric", "example": 36.6}
}


@app.get('/', response_class=HTMLResponse)
def index(request: Request):
    # Serve the static HTML file; the page will fetch /config to get feature list
    html_path = os.path.join(os.path.dirname(__file__), 'templates', 'index_fastapi.html')
    return FileResponse(html_path)


@app.on_event("startup")
async def start_llama_server():
    """On app startup, attempt to start the LLM server if not reachable and an executable is provided."""
    global LLAMA_PROCESS
    if not LLAMA_SERVER_URL or LLAMA_PROCESS:
        return
    parsed = urlparse(LLAMA_SERVER_URL)
    host = parsed.hostname or '127.0.0.1'
    port = parsed.port or (443 if parsed.scheme == 'https' else 80)

    if is_port_open(host, port):
        return

    if LLAMA_SERVER_EXE and os.path.exists(LLAMA_SERVER_EXE):
        cmd = [LLAMA_SERVER_EXE]
        if LLAMA_SERVER_ARGS:
            try:
                cmd += shlex.split(LLAMA_SERVER_ARGS)
            except Exception:
                cmd += [LLAMA_SERVER_ARGS]
        if LLAMA_MODEL_PATH:
            # common flag name; user can override with LLAMA_SERVER_ARGS
            cmd += ['--model', LLAMA_MODEL_PATH]
        # ensure port flag present
        if str(port) not in ' '.join(cmd):
            cmd += ['--http-port', str(port)]

        try:
            LLAMA_PROCESS = subprocess.Popen(cmd, cwd=os.path.dirname(LLAMA_SERVER_EXE))
        except Exception:
            LLAMA_PROCESS = None

        # wait briefly for server to be reachable
        for _ in range(20):
            if is_port_open(host, port):
                break
            time.sleep(0.5)


@app.on_event("shutdown")
async def stop_llama_server():
    """Terminate the spawned LLM server on shutdown if we started it."""
    global LLAMA_PROCESS
    if LLAMA_PROCESS:
        try:
            LLAMA_PROCESS.terminate()
            LLAMA_PROCESS.wait(timeout=5)
        except Exception:
            try:
                LLAMA_PROCESS.kill()
            except Exception:
                pass
        LLAMA_PROCESS = None


@app.get('/config')
def config():
    # Return structured metadata for each feature (name, type, example, options)
    # Expose a trimmed list for the UI: do not ask for `gender` (hidden)
    ui_features = [f for f in FEATURES if f != 'gender']
    result = []
    for f in ui_features:
        meta = FEATURE_CONFIG.get(f, {})
        item = {"name": f, "type": meta.get('type', 'numeric'), "example": meta.get('example')}
        if meta.get('options'):
            item['options'] = meta['options']
        result.append(item)
    return JSONResponse({"features": result})


def call_llm(prompt: str, timeout: int = 6) -> str | None:
    """Try to call a local LLM server (configured via .env). Return text or None."""
    if not LLAMA_SERVER_URL:
        return None
    url = LLAMA_SERVER_URL.rstrip('/') + LLAMA_API_PATH
    payload = {"prompt": prompt, "max_tokens": 300}
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        if not resp.ok:
            return None
        j = resp.json()
        # try common response shapes
        if isinstance(j, dict):
            if 'text' in j:
                return j['text']
            if 'output' in j:
                return j['output']
            if 'results' in j and isinstance(j['results'], list) and len(j['results'])>0:
                # LocalAI /v1/generate style
                try:
                    content = j['results'][0].get('content', [])
                    if content and isinstance(content, list):
                        for c in content:
                            if isinstance(c, dict) and 'text' in c:
                                return c['text']
                        # fallback to first item's text-like field
                        first = content[0]
                        if isinstance(first, dict):
                            for k in ('text','output'):
                                if k in first:
                                    return first[k]
                except Exception:
                    pass
        # fallback: try to extract any string
        txt = str(j)
        return txt
    except Exception:
        return None


def generate_lifestyle_suggestions(prediction: int, contrib_df: pd.DataFrame) -> str:
    """Create a user-friendly, non-medical suggestions string using SHAP contributions.
    Try LLM first; fall back to rule-based suggestions.
    """
    top_pos = contrib_df[contrib_df['contribution']>0].sort_values('contribution', ascending=False).head(3)
    top_neg = contrib_df[contrib_df['contribution']<0].sort_values('contribution').head(3)

    summary_lines = []
    summary_lines.append(f"Model prediction: {'Stroke (1)' if prediction==1 else 'No Stroke (0)'}.")
    if not contrib_df.empty:
        summary_lines.append('Top contributing features:')
        for _, r in pd.concat([top_pos, top_neg]).iterrows():
            summary_lines.append(f"- {r['feature']}: {r['contribution']:.3f}")

    prompt = (
        "You are a helpful assistant. Given the model prediction and the top contributing features (with numeric contributions), "
        "provide short, non-medical lifestyle suggestions the user can consider (e.g., diet, exercise, sleep, quitting smoking). "
        "Make clear these are general suggestions and not medical advice.\n\n"
        "Context:\n" + '\n'.join(summary_lines) + "\n\nRespond with a concise list of 5 suggestions."
    )

    llm_out = call_llm(prompt)
    if llm_out:
        return llm_out

    # fallback rule-based suggestions
    suggestions = []
    # map simple feature-driven suggestions (non-medical)
    contrib_map = {r['feature']: r['contribution'] for _, r in contrib_df.iterrows()}
    if contrib_map.get('smoking_status', 0) > 0:
        suggestions.append('If you smoke, consider reducing smoking and seeking support to quit; it improves overall health.')
    if contrib_map.get('bmi', 0) > 0:
        suggestions.append('Consider balanced diet and regular physical activity to manage weight (e.g., 30 minutes walking daily).')
    if contrib_map.get('avg_glucose_level', 0) > 0:
        suggestions.append('Reduce intake of simple sugars and refined carbs; favor whole foods and regular meals.')
    if contrib_map.get('hypertension', 0) > 0 or contrib_map.get('heart_disease', 0) > 0:
        suggestions.append('Adopt regular moderate exercise (walking, cycling), and reduce high-sodium processed foods.')
    if not suggestions:
        suggestions = [
            'Maintain balanced diet with vegetables and whole grains.',
            'Aim for regular physical activity (e.g., 150 minutes/week moderate exercise).',
            'Ensure adequate sleep and stress-management (relaxation, walks).',
            'Avoid tobacco and limit alcohol.',
            'Keep routine check-ins with healthcare professionals as needed.'
        ]

    return ' '.join(suggestions)


def format_readable_prediction(prediction: int, probability: float | None) -> str:
    """Return a human-readable sentence summarizing the prediction and probability."""
    label = 'stroke' if prediction == 1 else 'no stroke'
    if probability is None:
        return f"The model predicts {label}, but probability is unavailable."

    p = float(probability)
    pct = round(p * 100, 1)
    if p >= 0.5:
        level = 'high'
    elif p >= 0.2:
        level = 'moderate'
    elif p >= 0.05:
        level = 'low'
    else:
        level = 'very low'

    return f"The model predicts {label} with an estimated probability of {pct}%. This indicates a {level} likelihood." 


@app.post('/predict')
async def predict(payload: dict):
    # Ensure all features present
    data = {}
    for f in FEATURES:
        if f in payload and payload[f] != '':
            # Use provided value
            try:
                data[f] = float(payload[f])
            except Exception:
                data[f] = payload[f]
        else:
            # If not provided, fall back to example/default from FEATURE_CONFIG when available
            meta = FEATURE_CONFIG.get(f, {})
            if 'example' in meta:
                data[f] = meta['example']
            else:
                return JSONResponse({"error": f"Missing field {f} and no default available"}, status_code=400)

    sample = pd.DataFrame([data])

    pred = int(PIPELINE.predict(sample)[0])
    proba = None
    try:
        proba = float(PIPELINE.predict_proba(sample)[0][1])
    except Exception:
        proba = None

    img_url = None
    try:
        if SCALER is not None:
            X_in = SCALER.transform(sample)
        else:
            X_in = sample.values

        explainer = shap.TreeExplainer(MODEL)
        shap_values = explainer.shap_values(X_in)

        if isinstance(shap_values, list):
            arr = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        else:
            arr = shap_values

        contribs = arr[0]

        contrib_df = pd.DataFrame({"feature": FEATURES, "contribution": contribs})
        contrib_df = contrib_df.sort_values(by='contribution')

        # generate user-friendly suggestions using SHAP contributions (LLM or fallback)
        suggestions_text = generate_lifestyle_suggestions(pred, contrib_df)

        fig, ax = plt.subplots(figsize=(7, 4))
        colors = ['green' if v > 0 else 'red' for v in contrib_df['contribution']]
        ax.barh(contrib_df['feature'], contrib_df['contribution'], color=colors)
        ax.set_xlabel('SHAP contribution')
        fig.tight_layout()

        img_name = f"shap_{uuid.uuid4().hex}.png"
        img_path = os.path.join(static_dir, img_name)
        fig.savefig(img_path)
        plt.close(fig)
        img_url = f"/static/{img_name}"
    except Exception:
        img_url = None

    return JSONResponse({
        "prediction": pred,
        "probability": round(proba, 6) if proba is not None else None,
        "shap_image": img_url,
        "input": data,
        "suggestions": suggestions_text,
        "readable_prediction": format_readable_prediction(pred, proba)
    })
