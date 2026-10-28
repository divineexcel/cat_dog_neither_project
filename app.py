"""Pawtrait — Cat, Dog, or Neither? A beautiful multi-image classifier."""

from __future__ import annotations

import base64
import csv
import io
import hashlib
import json
from html import escape
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError
from pillow_heif import register_heif_opener
from image_preprocessing import model_views
from evaluation import build_metrics, evaluate_model, model_sha256

ROOT        = Path(__file__).resolve().parent
MODEL_PATH  = ROOT / "models" / "classifier.keras"
METRICS_PATH = ROOT / "models" / "evaluation.json"
HERO_PATH   = ROOT / "cat_dog_hero.png"
TARGET_SIZE = (224, 224)
CLASSES     = ("cat", "dog", "neither")
MAX_FILE_SIZE = 10 * 1024 * 1024
MAX_IMAGES    = 600
BATCH_CHUNK   = 32

register_heif_opener()

st.set_page_config(
    page_title="Pawtrait — Cat, Dog, or Neither?",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ─────────────────────────────  STYLES  ──────────────────────────────────────

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&family=DM+Mono:wght@400;500&display=swap');

/* ── tokens ── */
:root {
  --green:    #1a3a2a;
  --green2:   #1e4832;
  --amber:    #f0b429;
  --amber2:   #e8a020;
  --paper:    #f8f7f2;
  --card:     #ffffff;
  --ink:      #111a14;
  --muted:    #6b7c6f;
  --line:     #e4e6dc;
  --cat-col:  #e07a3b;
  --dog-col:  #3b7ae0;
  --nei-col:  #8b5cf6;
  --radius:   20px;
}

/* ── base ── */
html, body, [data-testid="stAppViewContainer"] {
  background: var(--paper);
  font-family: 'Outfit', sans-serif;
  color: var(--ink);
}
.block-container { max-width:1200px; padding:1.5rem 2rem 4rem; }
header[data-testid="stHeader"] { background:transparent; }
#MainMenu, footer { visibility:hidden; }

/* ── navbar ── */
.navbar {
  display:flex; align-items:center; justify-content:space-between;
  padding:14px 0 18px; border-bottom:2px solid var(--green);
  margin-bottom:56px;
}
.nav-logo {
  display:flex; align-items:center; gap:10px;
  font-weight:800; font-size:22px; letter-spacing:-.5px; color:var(--ink);
}
.nav-logo-dot {
  width:36px; height:36px; border-radius:50%;
  background:var(--green); display:grid; place-items:center;
  font-size:18px;
}
.nav-links {
  display:flex; gap:28px;
  font:500 13px 'Outfit',sans-serif; color:var(--muted);
}
.nav-links a {
  color:var(--muted); text-decoration:none; font:500 13px 'Outfit',sans-serif;
  transition:.15s; padding-bottom:3px;
}
.nav-links a:hover { color:var(--ink); }
.nav-links a.active { color:var(--amber); font-weight:700; border-bottom:2px solid var(--amber); }
.nav-cta {
  background:var(--green); color:white;
  border-radius:999px; padding:9px 22px;
  font:600 13px 'Outfit',sans-serif; cursor:pointer;
  border:none; transition:.15s; text-decoration:none; display:inline-block;
}
.nav-cta:hover { background:var(--amber); color:var(--ink); }

/* ── hero ── */
.hero-wrap {
  display:flex; align-items:center; gap:48px;
  margin-bottom:64px;
}
.hero-text { flex:1.1; }
.hero-tag {
  display:inline-flex; align-items:center; gap:7px;
  border:1.5px solid var(--line); background:white;
  border-radius:999px; padding:6px 14px;
  font:600 11px 'DM Mono',monospace; color:var(--muted);
  letter-spacing:.08em; text-transform:uppercase; margin-bottom:24px;
}
.hero-tag b { color:var(--amber); font-size:14px; }
.hero-title {
  font:800 clamp(42px,5.5vw,72px)/1.0 'Outfit',sans-serif;
  letter-spacing:-.04em; margin:0 0 20px; color:var(--ink);
}
.hero-title em { color:var(--amber); font-style:italic; font-weight:900; }
.hero-title span { color:var(--green); }
.hero-copy {
  color:var(--muted); font-size:16px; line-height:1.7;
  max-width:460px; margin-bottom:32px;
}
.hero-badges {
  display:flex; gap:12px; flex-wrap:wrap;
}
.hero-badge {
  background:var(--green); color:white;
  border-radius:999px; padding:8px 18px;
  font:600 12px 'Outfit',sans-serif;
}
.hero-badge.amber {
  background:var(--amber); color:var(--ink);
}
.hero-badge.outline {
  background:transparent; border:1.5px solid var(--line);
  color:var(--muted);
}
.hero-img-wrap {
  flex:1; display:flex; justify-content:center;
  position:relative;
}
.hero-img-bg {
  width:min(400px,42vw); height:min(400px,42vw);
  border-radius:50%;
  background:radial-gradient(circle, #f0b42933 0%, #f0b42908 70%, transparent 100%);
  display:flex; align-items:center; justify-content:center;
  border:2.5px solid #f0b42940;
}
.hero-img-bg img {
  width:90%; height:90%; object-fit:contain; border-radius:50%;
}
.hero-floating-tag {
  position:absolute; background:var(--green); color:white;
  border-radius:12px; padding:8px 14px;
  font:600 11px 'DM Mono',monospace; letter-spacing:.06em;
  box-shadow:0 8px 24px #1a3a2a40;
}
.hero-floating-tag.tl { top:8%; left:2%; }
.hero-floating-tag.br { bottom:12%; right:2%; background:var(--amber); color:var(--ink); }

/* ── ticker strip ── */
.ticker-strip {
  background:var(--green); color:white;
  padding:14px 0; margin:0 0 64px;
  border-radius:16px; overflow:hidden;
  display:flex; gap:48px; white-space:nowrap;
  font:600 13px 'DM Mono',monospace; letter-spacing:.12em; text-transform:uppercase;
}
.ticker-strip span { color:var(--amber); margin:0 8px; }

/* ── upload section ── */
.section-eyebrow {
  color:var(--amber); font:600 11px 'DM Mono',monospace;
  letter-spacing:.16em; text-transform:uppercase;
  margin-bottom:10px; display:flex; align-items:center; gap:8px;
}
.section-eyebrow::before { content:'—'; color:var(--amber); }
.section-title {
  font:700 clamp(28px,3.5vw,44px)/1.1 'Outfit',sans-serif;
  letter-spacing:-.03em; margin:0 0 8px; color:var(--ink);
}
.section-title em { color:var(--amber); font-style:italic; }
.section-sub { color:var(--muted); font-size:15px; margin-bottom:32px; max-width:500px; }

.upload-card {
  background:var(--card); border:1.5px solid var(--line);
  border-radius:var(--radius); padding:28px;
  box-shadow:0 12px 40px rgba(26,58,42,.07);
}
.upload-card-head {
  display:flex; justify-content:space-between; align-items:center;
  margin-bottom:18px;
}
.upload-card-head strong { font-size:15px; font-weight:700; }
.upload-card-head small {
  color:var(--muted); font:400 11px 'DM Mono',monospace;
}

[data-testid="stFileUploader"] section {
  background:#f3f5ee; border:2px dashed #c8d1be;
  border-radius:14px; padding:28px 16px;
}
[data-testid="stFileUploader"] section > button {
  background:var(--amber) !important;
  border:none !important;
  color:var(--ink) !important;
  border-radius:12px !important;
  font:700 14px 'Outfit',sans-serif !important;
  padding:12px 28px !important;
  width:100% !important;
  cursor:pointer;
  transition:.15s ease;
  letter-spacing:.02em;
  box-shadow: 0 4px 14px rgba(240,180,41,.35) !important;
}
[data-testid="stFileUploader"] section > button:hover {
  background:var(--amber2) !important;
  transform:translateY(-1px);
  box-shadow: 0 6px 20px rgba(240,180,41,.45) !important;
}
[data-testid="stFileUploader"] small { color:white; }
/* uploader drag zone label */
[data-testid="stFileUploader"] section p {
  font:500 13px 'Outfit',sans-serif; color:white; margin-bottom:12px;
}
[data-testid="stFileUploader"] section label {
  color:white !important;
}
[data-testid="stFileUploader"] section span {
  color:white !important;
}

.empty-state {
  min-height:160px; display:flex; flex-direction:column;
  align-items:center; justify-content:center;
  background:#f3f5ee; border:1.5px solid #e0e3d6;
  border-radius:14px; color:var(--muted); gap:8px;
  margin-bottom:16px;
}
.empty-state-icon { font-size:36px; }
.empty-state-text {
  font:500 11px 'DM Mono',monospace; letter-spacing:.08em; text-transform:uppercase;
}

/* ── buttons ── */
.stButton > button {
  width:100%; background:var(--green); color:white;
  border:0; border-radius:12px; padding:14px 18px; min-height:52px;
  font:700 14px 'Outfit',sans-serif; transition:.18s ease;
  letter-spacing:.02em;
}
.stButton > button:hover {
  background:var(--green2); transform:translateY(-1px);
  box-shadow:0 8px 20px #1a3a2a35;
}
.stButton > button:disabled { background:#b0bab2; color:white; }

/* ── model-loading notice ── */
.model-loading-box {
  background:linear-gradient(135deg, #1a3a2a 0%, #1e4832 100%);
  color:white; border-radius:16px; padding:20px 24px;
  margin-bottom:24px; display:flex; align-items:center; gap:16px;
}
.model-loading-spinner {
  width:32px; height:32px; flex:none;
  border:3px solid rgba(255,255,255,.2);
  border-top-color:var(--amber);
  border-radius:50%;
  animation:spin 0.8s linear infinite;
}
@keyframes spin { to { transform:rotate(360deg); } }
.model-loading-text strong { display:block; font-size:15px; margin-bottom:3px; }
.model-loading-text span { font-size:12px; opacity:.7; font-family:'DM Mono',monospace; }

/* ── progress ── */
.stProgress > div > div > div { background:var(--amber) !important; }
.stProgress > div > div { background:#e4e6dc !important; border-radius:99px !important; }
.stProgress > div > div > div { border-radius:99px !important; }

/* ── stat cards ── */
.stats-row {
  display:flex; gap:16px; margin:40px 0 32px; flex-wrap:wrap;
}
.stat-card {
  flex:1; min-width:130px; border-radius:18px; padding:20px 22px;
  border:1.5px solid var(--line); background:var(--card);
  box-shadow:0 6px 20px rgba(26,58,42,.05);
}
.stat-num {
  font:800 clamp(30px,3.5vw,44px)/1 'Outfit',sans-serif;
  letter-spacing:-.04em; margin-bottom:6px;
}
.stat-lbl {
  color:var(--muted); font:600 10px 'DM Mono',monospace;
  letter-spacing:.1em; text-transform:uppercase;
}
.stat-card.total .stat-num { color:var(--green); }
.stat-card.cat   .stat-num { color:var(--cat-col); }
.stat-card.dog   .stat-num { color:var(--dog-col); }
.stat-card.nei   .stat-num { color:var(--nei-col); }
.stat-card.err   .stat-num { color:#dc2626; }

/* ── image gallery ── */
.gallery-section-title {
  font:700 22px 'Outfit',sans-serif; margin:0 0 24px; color:var(--ink);
}
.gallery-grid {
  display:grid;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap:20px;
  margin-bottom:40px;
}
.gallery-item {
  background:var(--card); border-radius:16px;
  border:1.5px solid var(--line); overflow:hidden;
  box-shadow:0 6px 20px rgba(26,58,42,.06);
  transition:transform .18s ease, box-shadow .18s ease;
}
.gallery-item:hover {
  transform:translateY(-3px);
  box-shadow:0 12px 32px rgba(26,58,42,.12);
}
.gallery-item img {
  width:100%; aspect-ratio:1/1; object-fit:cover; display:block;
}
.gallery-item-body { padding:12px 14px 14px; }
.gallery-filename {
  font:500 11px 'DM Mono',monospace; color:var(--muted);
  white-space:nowrap; overflow:hidden; text-overflow:ellipsis;
  margin-bottom:8px;
}
.gallery-badge {
  display:inline-block; padding:4px 12px; border-radius:999px;
  font:700 11px 'Outfit',sans-serif; letter-spacing:.06em;
  text-transform:uppercase; margin-bottom:8px;
}
.gallery-badge.cat     { background:#fdecd8; color:#b85a18; }
.gallery-badge.dog     { background:#dbeafe; color:#1d4ed8; }
.gallery-badge.neither { background:#ede9fe; color:#6d28d9; }
.gallery-badge.error   { background:#fee2e2; color:#991b1b; }
.gallery-conf {
  font:700 18px 'Outfit',sans-serif; color:var(--ink); margin-bottom:6px;
}
.gallery-mini-bars { display:flex; flex-direction:column; gap:4px; }
.gallery-mini-row { display:flex; align-items:center; gap:7px; }
.gallery-mini-label {
  font:400 10px 'DM Mono',monospace; color:var(--muted);
  width:44px; flex:none; text-transform:uppercase;
}
.gallery-mini-track {
  flex:1; height:4px; border-radius:99px; background:#eaece4; overflow:hidden;
}
.gallery-mini-fill { height:100%; border-radius:99px; }
.gallery-mini-fill.cat     { background:var(--cat-col); }
.gallery-mini-fill.dog     { background:var(--dog-col); }
.gallery-mini-fill.neither { background:var(--nei-col); }
.gallery-mini-val {
  font:500 10px 'DM Mono',monospace; color:var(--muted); width:32px; text-align:right;
}

/* ── how strip ── */
.how-strip {
  background:var(--green); border-radius:20px; padding:32px 36px;
  display:flex; gap:0; margin:16px 0 0;
}
.how-step {
  flex:1; display:flex; gap:14px; align-items:flex-start;
  padding:0 24px; border-right:1px solid rgba(255,255,255,.12);
}
.how-step:first-child { padding-left:0; }
.how-step:last-child  { border:0; }
.step-no {
  width:32px; height:32px; border-radius:50%; flex:none;
  background:var(--amber); color:var(--ink);
  display:grid; place-items:center;
  font:700 12px 'DM Mono',monospace;
}
.step-copy b  { display:block; font-size:14px; color:white; margin-bottom:4px; font-weight:700; }
.step-copy small { color:rgba(255,255,255,.55); font-size:12px; }

/* ── confusion matrix ── */
.confusion-panel {
  background:var(--card); border:1.5px solid var(--line);
  border-radius:24px; padding:32px 36px; margin:0 0 48px;
  box-shadow:0 12px 40px rgba(26,58,42,.06);
}
.confusion-table-wrap { width:100%; overflow-x:auto; margin-top:18px; }
.confusion-table {
  width:100%; min-width:420px; border-collapse:separate; border-spacing:0;
  border:1px solid var(--line); border-radius:14px; overflow:hidden;
}
.confusion-table th, .confusion-table td {
  padding:13px 16px; text-align:center; border-right:1px solid var(--line);
  border-bottom:1px solid var(--line); font:500 14px 'DM Mono',monospace;
}
.confusion-table thead th { background:#f3f5ee; color:var(--muted); font-size:11px; letter-spacing:.06em; text-transform:uppercase; }
.confusion-table tbody th { background:#fafaf7; text-align:left; font:700 13px Outfit,sans-serif; text-transform:capitalize; }
.confusion-table .correct-cell { background:#e6f4ec; color:#17613a; font-weight:700; }
.confusion-table tr:last-child th, .confusion-table tr:last-child td { border-bottom:0; }
.confusion-table th:last-child, .confusion-table td:last-child { border-right:0; }
.confusion-table .outcome-tp { color:#17613a; font-weight:700; }
.confusion-table .outcome-tn { color:#245c45; }
.confusion-table .outcome-fp, .confusion-table .outcome-fn { color:#b45309; }
.confusion-note { margin:10px 0 0; color:var(--muted); font-size:12px; line-height:1.5; }

/* ── footer ── */
.site-footer {
  display:flex; justify-content:space-between; align-items:center;
  color:var(--muted); font:400 11px 'DM Mono',monospace;
  letter-spacing:.05em; margin-top:48px;
  border-top:1.5px solid var(--line); padding-top:20px;
}
.site-footer b { color:var(--ink); }

/* ── download button ── */
[data-testid="stDownloadButton"] > button {
  background:var(--green) !important;
  border:none !important;
  color:white !important;
  border-radius:12px !important;
  font:700 14px 'Outfit',sans-serif !important;
  padding:14px 28px !important;
  min-height:52px !important;
  width:100% !important;
  white-space:nowrap !important;
  overflow:visible !important;
  text-overflow:unset !important;
  letter-spacing:.02em !important;
  transition:.15s ease !important;
  display:flex !important;
  align-items:center !important;
  justify-content:center !important;
  gap:8px !important;
}
[data-testid="stDownloadButton"] > button:hover {
  background:var(--green2) !important;
  transform:translateY(-1px) !important;
  box-shadow:0 8px 20px #1a3a2a35 !important;
  color:white !important;
}

@media(max-width:700px) {
  .hero-wrap { flex-direction:column; }
  .how-strip { flex-direction:column; gap:20px; }
  .how-step  { border:0; padding:0 !important; }
  .gallery-grid { grid-template-columns: repeat(auto-fill,minmax(150px,1fr)); gap:12px; }
}
</style>
"""


# ─────────────────────────────  HELPERS  ─────────────────────────────────────

def img_to_b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def pil_to_b64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


@st.cache_resource(show_spinner=False)
def _load_model(model_mtime_ns: int):
    """Load and cache the Keras model."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    import keras
    return keras.models.load_model(MODEL_PATH)


def load_model():
    # Invalidate Streamlit's cached model after train.py replaces the artifact.
    return _load_model(MODEL_PATH.stat().st_mtime_ns)


@st.cache_data(show_spinner=False)
def load_confusion_metrics(model_mtime_ns: int, metrics_mtime_ns: int) -> dict | None:
    """Read the model-matched report, or evaluate locally if validation data exists."""
    if not MODEL_PATH.is_file():
        return None

    try:
        if METRICS_PATH.is_file():
            report = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
            if report.get("model_sha256") == model_sha256(MODEL_PATH):
                return report

        val_dir = ROOT / "data" / "val"
        if not val_dir.is_dir():
            return None

        import keras
        model = keras.models.load_model(MODEL_PATH)
        val_ds = keras.utils.image_dataset_from_directory(
            str(val_dir), image_size=(224, 224), batch_size=32,
            label_mode="int", shuffle=False,
        )
        return evaluate_model(model, val_ds, val_ds.class_names)
    except Exception:
        return None


def prepare_image(uploaded_file) -> Image.Image:
    raw = uploaded_file.getvalue()
    if len(raw) > MAX_FILE_SIZE:
        raise ValueError("Image is over 10 MB.")
    try:
        with Image.open(io.BytesIO(raw)) as src:
            src.seek(0); src.load()
            try:
                oriented = ImageOps.exif_transpose(src)
            except (OSError, ValueError):
                oriented = src
            img = oriented.convert("RGB")
            img.load()
            return img.copy()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError("Could not decode image — try JPG, PNG, WebP, or HEIC.") from exc


def predict_batch(images: list[Image.Image]) -> list[np.ndarray]:
    import keras
    model = load_model()
    results = []
    for start in range(0, len(images), BATCH_CHUNK):
        chunk = images[start: start + BATCH_CHUNK]
        views = [view for img in chunk for view in model_views(img, TARGET_SIZE)]
        arrays = [np.asarray(view, dtype=np.float32) for view in views]
        # Preprocessing is embedded in the saved model. Pass raw RGB pixels,
        # matching the [0, 255] tensors used by the training input pipeline.
        batch = np.stack(arrays, axis=0)
        view_preds = model.predict(batch, verbose=0)
        cursor = 0
        for image in chunk:
            view_count = len(model_views(image, TARGET_SIZE))
            results.append(np.mean(view_preds[cursor:cursor + view_count], axis=0))
            cursor += view_count
    return results


def build_csv(rows: list[dict]) -> bytes:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=["filename","prediction","confidence_%","cat_%","dog_%","neither_%","error"])
    w.writeheader(); w.writerows(rows)
    return buf.getvalue().encode()


# ─────────────────────────────  MAIN  ────────────────────────────────────────

def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)

    # ── navbar ──────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="navbar">
      <div class="nav-logo">
        <div class="nav-logo-dot">🐾</div>
        <span>Pawtrait</span>
      </div>
      <div class="nav-links">
        <a href="#home" class="active">Home</a>
        <a href="#about">About</a>
        <a href="#performance">Confusion Matrix</a>
        <a href="#how-it-works">How it works</a>
        <a href="#results">Results</a>
      </div>
      <a href="#upload" class="nav-cta">Try it free ↓</a>
    </div>
    """, unsafe_allow_html=True)

    # ── hero ─────────────────────────────────────────────────────────────────
    hero_b64 = img_to_b64(HERO_PATH) if HERO_PATH.exists() else ""

    st.markdown(f"""
    <div id="home" class="hero-wrap">
      <div class="hero-text">
        <div class="hero-tag"><b>🐱</b> AI-Powered Pet Classifier <b>🐶</b></div>
        <h1 class="hero-title">
          Is it a <em>Cat,</em><br>
          a <span>Dog,</span><br>
          or Something Else?
        </h1>
        <p class="hero-copy">
          Upload up to 600 photos and let Pawtrait instantly tell you what's in each one.
          Our AI model has been trained on thousands of cat, dog, and "neither" images — 
          so it knows its paws from its claws.
        </p>
        <div class="hero-badges">
          <div class="hero-badge">🐱 Cat Detection</div>
          <div class="hero-badge amber">🐶 Dog Recognition</div>
          <div class="hero-badge outline">🌀 Up to 600 Images</div>
        </div>
      </div>
      <div class="hero-img-wrap">
        <div class="hero-img-bg">
          {'<img src="data:image/png;base64,' + hero_b64 + '" alt="cat and dog"/>' if hero_b64 else '<span style="font-size:80px">🐾</span>'}
        </div>
        <div class="hero-floating-tag tl">🐱 Cat · 97.3%</div>
        <div class="hero-floating-tag br">🐶 Dog · 94.8%</div>
      </div>
    </div>

    <!-- About section -->
    <div id="about" style="background:var(--card);border:1.5px solid var(--line);border-radius:24px;padding:40px 44px;margin-bottom:48px;">
      <div class="section-eyebrow" style="margin-bottom:12px;">About Pawtrait</div>
      <div style="display:flex;gap:48px;align-items:flex-start;flex-wrap:wrap;">
        <div style="flex:1.2;min-width:260px;">
          <h2 style="font:700 32px/1.1 Outfit,sans-serif;letter-spacing:-.03em;margin:0 0 16px;">What is <em style="color:var(--amber);font-style:italic;">Pawtrait?</em></h2>
          <p style="color:var(--muted);font-size:15px;line-height:1.75;margin:0 0 16px;">
            Pawtrait is an AI-powered image classifier built with a fine-tuned <strong>MobileNetV2</strong> model 
            trained on thousands of real cat, dog, and non-animal photos. It classifies images into three categories:
          </p>
          <div style="display:flex;gap:12px;flex-wrap:wrap;margin-bottom:16px;">
            <div style="background:#fdecd8;color:#b85a18;padding:8px 16px;border-radius:999px;font:700 13px Outfit,sans-serif;">🐱 Cat</div>
            <div style="background:#dbeafe;color:#1d4ed8;padding:8px 16px;border-radius:999px;font:700 13px Outfit,sans-serif;">🐶 Dog</div>
            <div style="background:#ede9fe;color:#6d28d9;padding:8px 16px;border-radius:999px;font:700 13px Outfit,sans-serif;">🌀 Neither</div>
          </div>
          <p style="color:var(--muted);font-size:15px;line-height:1.75;margin:0;">
            Upload a single image or batch up to <strong>600 at once</strong>. The model processes them in parallel 
            and returns every result instantly — complete with confidence scores and a CSV export.
          </p>
        </div>
        <div style="flex:1;min-width:220px;display:flex;flex-direction:column;gap:14px;">
          <div style="background:var(--paper);border-radius:16px;padding:18px 20px;border:1.5px solid var(--line);"><b style="color:var(--green);font-size:22px;">MobileNetV2</b><br><span style="color:var(--muted);font-size:13px;">Pre-trained backbone, fine-tuned on 3,000 images</span></div>
          <div style="background:var(--paper);border-radius:16px;padding:18px 20px;border:1.5px solid var(--line);"><b style="color:var(--amber);font-size:22px;">600 Images</b><br><span style="color:var(--muted);font-size:13px;">Process up to 600 photos in a single batch</span></div>
          <div style="background:var(--paper);border-radius:16px;padding:18px 20px;border:1.5px solid var(--line);"><b style="color:var(--nei-col);font-size:22px;">3 Classes</b><br><span style="color:var(--muted);font-size:13px;">Cat, Dog, or Neither — always a clear verdict</span></div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── validation confusion matrix ─────────────────────────────────────────
    metrics = load_confusion_metrics(
        MODEL_PATH.stat().st_mtime_ns if MODEL_PATH.is_file() else 0,
        METRICS_PATH.stat().st_mtime_ns if METRICS_PATH.is_file() else 0,
    )
    if metrics:
        names = metrics["class_names"]
        matrix = metrics["confusion_matrix"]
        total = metrics["n_val"]
        outcome_rows = []
        for index, name in enumerate(names):
            tp = matrix[index][index]
            fp = sum(matrix[actual][index] for actual in range(len(names)) if actual != index)
            fn = sum(matrix[index][predicted] for predicted in range(len(names)) if predicted != index)
            tn = total - tp - fp - fn
            outcome_rows.append(
                "<tr><th>{label}</th><td class='outcome-tp'>{tp}</td>"
                "<td class='outcome-tn'>{tn}</td><td class='outcome-fp'>{fp}</td>"
                "<td class='outcome-fn'>{fn}</td></tr>".format(
                    label=escape(name), tp=tp, tn=tn, fp=fp, fn=fn,
                )
            )
        rows = "".join(outcome_rows)
        st.markdown(f"""
        <div id="performance" class="confusion-panel">
          <div class="section-eyebrow">Model Performance</div>
          <h2 style="font:700 30px/1.1 Outfit,sans-serif;letter-spacing:-.03em;margin:0 0 8px;">
            Validation <em style="color:var(--amber);font-style:italic;">Confusion Matrix Outcomes</em>
          </h2>
          <p style="color:var(--muted);font-size:14px;margin:0;">
            One-versus-rest counts for each class, calculated from {metrics['n_val']} validation images.
          </p>
          <div class="confusion-table-wrap">
            <table class="confusion-table">
              <thead><tr><th>Class</th><th>TP</th><th>TN</th><th>FP</th><th>FN</th></tr></thead>
              <tbody>{rows}</tbody>
            </table>
          </div>
          <p class="confusion-note"><b>TP</b> true positive · <b>TN</b> true negative · <b>FP</b> false positive · <b>FN</b> false negative. For each row, that class is positive and the other two classes are grouped as negative.</p>
        </div>
        """, unsafe_allow_html=True)

    # ── ticker strip ─────────────────────────────────────────────────────────
    st.markdown("""
    <div class="ticker-strip">
      &nbsp;&nbsp;&nbsp;Cat Detection <span>✦</span>
      Dog Recognition <span>✦</span>
      Batch Processing <span>✦</span>
      600 Images at Once <span>✦</span>
      CSV Export <span>✦</span>
      AI-Powered <span>✦</span>
      3 Classes <span>✦</span>
      Cat Detection <span>✦</span>
      Dog Recognition <span>✦</span>
      Batch Processing <span>✦</span>
    </div>
    """, unsafe_allow_html=True)

    # ── upload section ───────────────────────────────────────────────────────
    st.markdown('<div id="upload"></div>', unsafe_allow_html=True)
    ul, ur = st.columns([1, 1.1], gap="large")

    with ul:
        st.markdown("""
        <div class="section-eyebrow">Upload Images</div>
        <h2 class="section-title">Drop Your Photos<br>Get <em>Instant Verdicts</em></h2>
        <p class="section-sub">
          Select up to 600 images at once. Our model runs them in fast batches
          and shows you every result — with a beautiful gallery beneath.
        </p>
        <p style="color:var(--muted);font-size:13px;line-height:1.6;">
          ✅ Supports JPG, PNG, WebP, HEIC, BMP<br>
          ✅ Up to 10 MB per image · Up to 600 images<br>
          ✅ Results appear as a visual gallery below
        </p>
        """, unsafe_allow_html=True)

    with ur:
        st.markdown("""
        <div class="upload-card">
          <div class="upload-card-head">
            <strong>🐾 Upload your images</strong>
            <small>JPG · PNG · WEBP · HEIC · up to 600</small>
          </div>
          <p style="color:var(--muted);font-size:13px;margin:0 0 16px;"
          >Click the button below to browse and select your photos — or drag &amp; drop them directly into the box.</p>
        """, unsafe_allow_html=True)

        uploaded_files = st.file_uploader(
            "📂  Browse &amp; select your images — or drag &amp; drop here",
            type=["jpg","jpeg","png","webp","bmp","heic","heif"],
            accept_multiple_files=True,
            label_visibility="collapsed",
            help=f"Select up to {MAX_IMAGES} images, each under 10 MB.",
        )

        if len(uploaded_files) > MAX_IMAGES:
            st.warning(f"Only the first {MAX_IMAGES} images will be processed.")
            uploaded_files = uploaded_files[:MAX_IMAGES]

        n = len(uploaded_files)

        if n == 0:
            st.markdown("""
            <div class="empty-state">
              <div class="empty-state-icon">🐾</div>
              <div class="empty-state-text">Your images will appear here</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div style="color:#245c45;font:600 13px Outfit,sans-serif;'
                f'background:#e6f4ec;border-radius:10px;padding:10px 14px;margin:0 0 12px;">'
                f'✅ {n} image{"s" if n>1 else ""} ready to classify</div>',
                unsafe_allow_html=True
            )

        analyze = st.button(
            f"🔍  Classify {n} image{'s' if n != 1 else ''} now  →" if n > 0 else "🔍  Select images above to classify",
            type="primary",
            disabled=n == 0,
            use_container_width=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

    # ── model loading + inference ─────────────────────────────────────────────
    if analyze and n > 0:
        # Step 1: visible model loading bar
        model_status = st.empty()
        model_status.markdown("""
        <div class="model-loading-box">
          <div class="model-loading-spinner"></div>
          <div class="model-loading-text">
            <strong>Loading AI model…</strong>
            <span>Warming up MobileNetV2 · This happens only once</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        try:
            load_model()
        except FileNotFoundError:
            model_status.error("❌ Model not found at `models/classifier.keras`. Please train the model first.")
            return

        model_status.markdown("""
        <div class="model-loading-box" style="background:linear-gradient(135deg,#1e4832,#245c45);">
          <div style="font-size:24px">✅</div>
          <div class="model-loading-text">
            <strong>Model ready!</strong>
            <span>MobileNetV2 loaded and ready to classify</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        import time; time.sleep(0.6)
        model_status.empty()

        # Step 2: load + decode images with progress
        progress = st.progress(0, text="Loading images…")
        images_ok: list[tuple[str, Image.Image]] = []
        errors: list[tuple[str, str]] = []

        for i, f in enumerate(uploaded_files):
            progress.progress((i + 1) / n, text=f"📂  Loading {i + 1} of {n} — {f.name}")
            try:
                images_ok.append((f.name, prepare_image(f)))
            except ValueError as exc:
                errors.append((f.name, str(exc)))

        # Step 3: batch inference
        progress.progress(0.98, text=f"🧠  Running model on {len(images_ok)} images…")
        all_scores = predict_batch([img for _, img in images_ok])
        progress.empty()

        # Build result rows
        rows: list[dict] = []
        for (fname, _), scores in zip(images_ok, all_scores):
            best = int(np.argmax(scores))
            rows.append({
                "filename":     fname,
                "prediction":   CLASSES[best],
                "confidence_%": round(float(scores[best]) * 100, 1),
                "cat_%":        round(float(scores[0]) * 100, 1),
                "dog_%":        round(float(scores[1]) * 100, 1),
                "neither_%":    round(float(scores[2]) * 100, 1),
                "error":        "",
                "_pil":         [img for _, img in images_ok][[fname2 for fname2, _ in images_ok].index(fname)],
            })
        for fname, err in errors:
            rows.append({
                "filename":"", "prediction":"error",
                "confidence_%":0, "cat_%":0, "dog_%":0, "neither_%":0,
                "error": err, "_pil": None,
            })

        # persist
        st.session_state["pawtrait_batch"] = {
            "key": hashlib.sha256("".join(f.name for f in uploaded_files).encode()).hexdigest(),
            "rows": [{k: v for k, v in r.items() if k != "_pil"} for r in rows],
            "pils": [r["_pil"] for r in rows],
            "names": [r["filename"] for r in rows],
        }

    # ── results ───────────────────────────────────────────────────────────────
    batch = st.session_state.get("pawtrait_batch")
    if batch and n > 0:
        rows    = batch["rows"]
        pils    = batch["pils"]
        ok_rows = [r for r in rows if r["prediction"] != "error"]
        counts  = {c: sum(1 for r in ok_rows if r["prediction"] == c) for c in CLASSES}
        total   = len(ok_rows)
        err_cnt = len(rows) - total

        # ── stat cards ──────────────────────────────────────────────────────
        err_html = f'<div class="stat-card err"><div class="stat-num">{err_cnt}</div><div class="stat-lbl">⚠ Errors</div></div>' if err_cnt else ""
        st.markdown(f"""
        <div class="stats-row">
          <div class="stat-card total"><div class="stat-num">{total}</div><div class="stat-lbl">Total Classified</div></div>
          <div class="stat-card cat"><div class="stat-num">{counts["cat"]}</div><div class="stat-lbl">🐱 Cats</div></div>
          <div class="stat-card dog"><div class="stat-num">{counts["dog"]}</div><div class="stat-lbl">🐶 Dogs</div></div>
          <div class="stat-card nei"><div class="stat-num">{counts["neither"]}</div><div class="stat-lbl">🌀 Neither</div></div>
          {err_html}
        </div>
        """, unsafe_allow_html=True)

        # ── CSV download ─────────────────────────────────────────────────────
        csv_bytes = build_csv(rows)
        st.markdown('<div id="results"></div>', unsafe_allow_html=True)
        st.download_button(
            label=f"⬇  Download all {total} results as CSV  (pawtrait_results.csv)",
            data=csv_bytes,
            file_name="pawtrait_results.csv",
            mime="text/csv",
            use_container_width=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── IMAGE GALLERY with classification beneath ────────────────────────
        st.markdown('<div class="gallery-section-title">📸 Image Gallery &amp; Results</div>', unsafe_allow_html=True)

        COLS = 4
        col_groups = [rows[i: i + COLS] for i in range(0, len(rows), COLS)]
        pil_groups = [pils[i: i + COLS] for i in range(0, len(pils), COLS)]

        for row_items, pil_items in zip(col_groups, pil_groups):
            cols = st.columns(len(row_items), gap="medium")
            for col, r, pil in zip(cols, row_items, pil_items):
                with col:
                    pred   = r["prediction"]
                    conf   = r["confidence_%"]
                    cat_p  = r["cat_%"]
                    dog_p  = r["dog_%"]
                    nei_p  = r["neither_%"]
                    fname  = r["filename"] or r.get("error","error")

                    if pil is not None:
                        # thumbnail: square crop
                        thumb = pil.copy()
                        side  = min(thumb.size)
                        l     = (thumb.width  - side) // 2
                        t     = (thumb.height - side) // 2
                        thumb = thumb.crop((l, t, l + side, t + side)).resize((300, 300), Image.Resampling.LANCZOS)
                        b64   = pil_to_b64(thumb)

                        mini_bars = f"""
                        <div class="gallery-mini-bars">
                          <div class="gallery-mini-row">
                            <span class="gallery-mini-label">Cat</span>
                            <div class="gallery-mini-track"><div class="gallery-mini-fill cat" style="width:{cat_p}%"></div></div>
                            <span class="gallery-mini-val">{cat_p:.0f}%</span>
                          </div>
                          <div class="gallery-mini-row">
                            <span class="gallery-mini-label">Dog</span>
                            <div class="gallery-mini-track"><div class="gallery-mini-fill dog" style="width:{dog_p}%"></div></div>
                            <span class="gallery-mini-val">{dog_p:.0f}%</span>
                          </div>
                          <div class="gallery-mini-row">
                            <span class="gallery-mini-label">Neither</span>
                            <div class="gallery-mini-track"><div class="gallery-mini-fill neither" style="width:{nei_p}%"></div></div>
                            <span class="gallery-mini-val">{nei_p:.0f}%</span>
                          </div>
                        </div>
                        """

                        st.markdown(f"""
                        <div class="gallery-item">
                          <img src="data:image/jpeg;base64,{b64}" alt="{fname}"/>
                          <div class="gallery-item-body">
                            <div class="gallery-filename" title="{fname}">{fname}</div>
                            <div class="gallery-badge {pred}">{pred}</div>
                            <div class="gallery-conf">{conf:.1f}%</div>
                            {mini_bars}
                          </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="gallery-item">
                          <div style="height:160px;background:#fee2e2;display:flex;align-items:center;justify-content:center;font-size:32px">⚠️</div>
                          <div class="gallery-item-body">
                            <div class="gallery-filename">{fname}</div>
                            <div class="gallery-badge error">error</div>
                            <div style="font-size:11px;color:var(--muted);margin-top:6px">{r.get("error","")}</div>
                          </div>
                        </div>
                        """, unsafe_allow_html=True)

        # A confusion matrix needs ground-truth labels as well as predictions.
        # Let the user label each upload, then calculate a batch-specific matrix.
        st.markdown('<div id="upload-confusion"></div>', unsafe_allow_html=True)
        st.markdown("### Confusion matrix for these uploads")
        st.caption(
            "The model supplies predictions. Select each image’s actual class below; "
            "only labeled images are included in the live matrix."
        )
        label_options = ["Select actual class", *CLASSES]
        labeled_true, labeled_pred = [], []
        with st.expander("Label uploaded images", expanded=True):
            label_cols = st.columns(2)
            for index, row in enumerate(rows):
                if row["prediction"] == "error":
                    continue
                with label_cols[index % 2]:
                    actual = st.selectbox(
                        f"{row['filename']} · model predicted {row['prediction']}",
                        options=label_options,
                        key=f"actual_label_{batch['key']}_{index}",
                    )
                if actual in CLASSES:
                    labeled_true.append(CLASSES.index(actual))
                    labeled_pred.append(CLASSES.index(row["prediction"]))

        if labeled_true:
            upload_metrics = build_metrics(labeled_true, labeled_pred, list(CLASSES))
            matrix = upload_metrics["confusion_matrix"]
            names = upload_metrics["class_names"]
            st.caption(
                f"Showing {upload_metrics['n_val']} labeled image(s). "
                f"{total - upload_metrics['n_val']} classified image(s) still need an actual label."
            )
            matrix_headers = "".join(f"<th>{escape(name)}</th>" for name in names)
            matrix_rows = "".join(
                "<tr><th>{actual}</th>{cells}</tr>".format(
                    actual=escape(actual),
                    cells="".join(
                        '<td class="{style}">{count}</td>'.format(
                            style="correct-cell" if row_index == col_index else "",
                            count=count,
                        )
                        for col_index, count in enumerate(row)
                    ),
                )
                for row_index, (actual, row) in enumerate(zip(names, matrix))
            )
            st.markdown(f"""
            <div class="confusion-table-wrap">
              <table class="confusion-table">
                <thead><tr><th>Actual ↓ / Predicted →</th>{matrix_headers}</tr></thead>
                <tbody>{matrix_rows}</tbody>
              </table>
            </div>
            """, unsafe_allow_html=True)

            outcome_rows = []
            for index, name in enumerate(names):
                tp = matrix[index][index]
                fp = sum(matrix[actual][index] for actual in range(len(names)) if actual != index)
                fn = sum(matrix[index][predicted] for predicted in range(len(names)) if predicted != index)
                tn = upload_metrics["n_val"] - tp - fp - fn
                outcome_rows.append(
                    "<tr><th>{label}</th><td>{tp}</td><td>{tn}</td><td>{fp}</td><td>{fn}</td></tr>".format(
                        label=escape(name), tp=tp, tn=tn, fp=fp, fn=fn,
                    )
                )
            st.markdown(f"""
            <div class="confusion-table-wrap">
              <table class="confusion-table">
                <thead><tr><th>Class (one vs rest)</th><th>TP</th><th>TN</th><th>FP</th><th>FN</th></tr></thead>
                <tbody>{''.join(outcome_rows)}</tbody>
              </table>
            </div>
            <p class="confusion-note">TP = true positive · TN = true negative · FP = false positive · FN = false negative.</p>
            """, unsafe_allow_html=True)
        else:
            st.info("Select at least one actual class to build the confusion matrix for this upload batch.")

    # ── how it works ──────────────────────────────────────────────────────────
    st.markdown("""
    <div id="how-it-works" class="how-strip">
      <div class="how-step">
        <div class="step-no">01</div>
        <div class="step-copy">
          <b>Pick your photos</b>
          <small>Click "Browse" or drag &amp; drop — up to 600 images at once (JPG, PNG, WebP, HEIC)</small>
        </div>
      </div>
      <div class="how-step">
        <div class="step-no">02</div>
        <div class="step-copy">
          <b>AI takes a look</b>
          <small>MobileNetV2 loads (once), then classifies your images in fast parallel batches</small>
        </div>
      </div>
      <div class="how-step">
        <div class="step-no">03</div>
        <div class="step-copy">
          <b>See every verdict</b>
          <small>A visual gallery with Cat / Dog / Neither badges, confidence bars, and a full CSV export</small>
        </div>
      </div>
    </div>
    <div class="site-footer">
      <span><b>Pawtrait</b> · AI Pet Classifier · v2.0</span>
      <span>CAT &nbsp;/&nbsp; DOG &nbsp;/&nbsp; NEITHER</span>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
