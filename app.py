from __future__ import annotations

from pathlib import Path

import streamlit as st
from PIL import Image

from src.inference import default_checkpoint_path, load_config, load_model_for_inference, predict_image
from src.utils import load_json


st.set_page_config(
    page_title="Plant Disease Detection Using CNN",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --page-bg: #eef3e8;
        --panel-bg: rgba(255, 255, 255, 0.95);
        --text-main: #162012;
        --text-muted: #566253;
        --accent: #2f6b3b;
    }
    .stApp {
        background: radial-gradient(circle at top left, #f3f8ee 0%, #edf4ea 35%, #f8f7f2 100%);
        color: var(--text-main);
    }
    .stApp, .main, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        color: var(--text-main) !important;
    }
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
    .stApp p, .stApp span, .stApp label, .stApp li, .stApp small,
    .stApp div[data-testid="stMarkdownContainer"] {
        color: var(--text-main);
    }
    .hero {
        padding: 1.6rem 1.8rem;
        border-radius: 24px;
        background: linear-gradient(135deg, rgba(31, 71, 46, 0.98), rgba(69, 104, 55, 0.96));
        color: white;
        box-shadow: 0 14px 40px rgba(30, 48, 28, 0.18);
        margin-bottom: 1rem;
    }
    .hero h1, .hero p, .hero * {
        color: white !important;
    }
    .hero h1 {
        margin: 0;
        font-size: 2.1rem;
        line-height: 1.1;
    }
    .hero p {
        margin: 0.5rem 0 0;
        opacity: 0.88;
        font-size: 1rem;
    }
    .card {
        background: var(--panel-bg);
        color: var(--text-main);
        border: 1px solid rgba(32, 61, 30, 0.12);
        border-radius: 18px;
        padding: 1rem 1rem 0.75rem 1rem;
        box-shadow: 0 10px 30px rgba(28, 44, 23, 0.08);
    }
    .stCaption, .st-emotion-cache-1y4p8pa, .stMarkdown p {
        color: var(--text-muted) !important;
    }
    [data-testid="stMetric"] {
        background: var(--panel-bg);
        border: 1px solid rgba(32, 61, 30, 0.12);
        border-radius: 18px;
        padding: 0.8rem 1rem;
        box-shadow: 0 10px 30px rgba(28, 44, 23, 0.08);
    }
    [data-testid="stMetric"] * {
        color: var(--text-main) !important;
    }
    [data-testid="stAlert"] {
        color: var(--text-main) !important;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1d2220 0%, #262d29 100%);
    }
    [data-testid="stSidebar"] * {
        color: #f3f6f0 !important;
    }
    [data-testid="stSidebar"] .stCaption,
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] span {
        color: #d9e2d4 !important;
    }
    [data-testid="stSidebar"] input,
    [data-testid="stSidebar"] textarea,
    [data-testid="stSidebar"] .stTextInput input,
    [data-testid="stSidebar"] .stSelectbox div,
    [data-testid="stSidebar"] div[role="slider"] {
        color: #f3f6f0 !important;
        background: rgba(255, 255, 255, 0.06) !important;
        border-color: rgba(255, 255, 255, 0.12) !important;
    }
    .stTabs [data-baseweb="tab-list"] button {
        color: var(--text-muted) !important;
        background: transparent !important;
    }
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        color: var(--text-main) !important;
        border-bottom: 2px solid var(--accent) !important;
    }
    [data-testid="stFileUploader"] {
        background: var(--panel-bg) !important;
        border: 1px solid rgba(32, 61, 30, 0.14) !important;
        border-radius: 18px !important;
        padding: 0.35rem !important;
    }
    [data-testid="stFileUploader"] * {
        color: var(--text-main) !important;
    }
    [data-testid="stFileUploaderDropzone"] {
        background: rgba(47, 107, 59, 0.06) !important;
        border-color: rgba(47, 107, 59, 0.24) !important;
    }
    [data-testid="stFileUploaderDropzone"] button {
        color: #ffffff !important;
        background: var(--accent) !important;
        border: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


ROOT = Path(__file__).resolve().parent
CONFIG_OPTIONS = {
    "ResNet-18": str(ROOT / "configs" / "resnet18.yaml"),
    "EfficientNet-B0": str(ROOT / "configs" / "efficientnet_b0.yaml"),
}


def has_filtered_labels() -> bool:
    label_map_path = ROOT / "data" / "splits" / "label_map.json"
    if not label_map_path.exists():
        return False

    try:
        label_map = load_json(label_map_path)
    except Exception:
        return False

    return not any(str(name).lower().startswith("tomato") for name in label_map.keys())


def latest_trained_model_name() -> str | None:
    results_path = ROOT / "reports" / "results.json"
    if not results_path.exists():
        return None

    try:
        results = load_json(results_path)
    except Exception:
        return None

    if int(results.get("num_classes", 0) or 0) == 5 and has_filtered_labels():
        best_checkpoint = Path(str(results.get("best_checkpoint", "")))
        if best_checkpoint.exists():
            for model_name, config_path in CONFIG_OPTIONS.items():
                cfg = load_config(config_path)
                exp_name = str(cfg.get("experiment", {}).get("name", "")).lower().strip()
                if exp_name and best_checkpoint.stem.lower().startswith(exp_name):
                    return model_name

    best_checkpoint = Path(str(results.get("best_checkpoint", "")))
    best_stem = best_checkpoint.stem.lower()

    for model_name, config_path in CONFIG_OPTIONS.items():
        cfg = load_config(config_path)
        exp_name = str(cfg.get("experiment", {}).get("name", "")).lower().strip()
        if exp_name and best_stem.startswith(exp_name):
            return model_name

    return None


def available_model_options() -> dict[str, str]:
    available: dict[str, str] = {}
    for model_name, config_path in CONFIG_OPTIONS.items():
        cfg = load_config(config_path)
        checkpoint_path = default_checkpoint_path(cfg)
        if checkpoint_path.exists() and (has_filtered_labels() or int(cfg.get("model", {}).get("num_classes", 0) or 0) == 5):
            available[model_name] = config_path
    return available or CONFIG_OPTIONS


def checkpoint_help_text(config_path: str) -> str:
    cfg = load_config(config_path)
    suggested_checkpoint = default_checkpoint_path(cfg)
    available = sorted((ROOT / "reports" / "checkpoints").glob("*_best.pt"))
    available_text = ", ".join(path.name for path in available) if available else "none"
    if suggested_checkpoint.exists():
        status = f"Suggested: {suggested_checkpoint}"
    else:
        status = f"No trained checkpoint found for {Path(config_path).stem}"
    return f"{status}\nAvailable in reports/checkpoints: {available_text}"


def artifact_cache_key(config_path: str, checkpoint_path: str | None) -> str:
    paths = [Path(config_path).expanduser().resolve()]
    if checkpoint_path:
        paths.append(Path(checkpoint_path).expanduser().resolve())

    label_map_path = ROOT / "data" / "splits" / "label_map.json"
    if label_map_path.exists():
        paths.append(label_map_path)

    parts: list[str] = []
    for path in paths:
        try:
            stat = path.stat()
            parts.append(f"{path}:{stat.st_mtime_ns}:{stat.st_size}")
        except OSError:
            parts.append(str(path))
    return "|".join(parts)


@st.cache_resource(show_spinner=False)
def load_artifacts(config_path: str, checkpoint_path: str | None, cache_key: str):
    _ = cache_key
    return load_model_for_inference(config_path, checkpoint_path=checkpoint_path)


def render_prediction(result: dict) -> None:
    top = result["top_prediction"]
    st.metric("Prediction", top["label"], f"{top['confidence'] * 100:.1f}% confidence")

    rows = result["predictions"]
    st.markdown("#### Top predictions")
    for row in rows:
        st.progress(float(row["confidence"]))
        st.write(f"{row['rank']}. {row['label']} - {row['confidence'] * 100:.2f}%")


def main() -> None:
    st.markdown(
        """
        <div class="hero">
            <h1>Plant Disease Prediction</h1>
            <p>Upload a leaf image or use your camera. The model will return the predicted disease class and confidence.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Model Settings")
        model_options = available_model_options()
        model_names = list(model_options.keys())
        default_model_name = latest_trained_model_name()
        default_index = model_names.index(default_model_name) if default_model_name in model_names else 0
        model_name = st.selectbox("Choose a model", model_names, index=default_index)
        config_path = model_options[model_name]
        cfg = load_config(config_path)
        suggested_checkpoint = default_checkpoint_path(cfg)
        checkpoint_value = str(suggested_checkpoint) if suggested_checkpoint.exists() else ""
        checkpoint_override = st.text_input("Optional checkpoint path", value=checkpoint_value)
        top_k = st.slider("Top-K predictions", min_value=1, max_value=5, value=3, step=1)
        st.caption("Train the model first, then use this page to test it on single images.")
        st.caption(checkpoint_help_text(config_path))

    checkpoint_path = checkpoint_override.strip() or None

    if checkpoint_path is None and not has_filtered_labels():
        st.warning("The current label map still contains tomato classes. Refresh data/splits/label_map.json before using the app.")

    if checkpoint_path is None and not default_checkpoint_path(cfg).exists():
        st.error("No trained checkpoint exists for the selected model yet.")
        st.stop()

    try:
        model, cfg, label_names, device, ckpt_path = load_artifacts(
            config_path,
            checkpoint_path,
            artifact_cache_key(config_path, checkpoint_path),
        )
    except Exception as exc:
        st.error(str(exc))
        st.info("Run the training pipeline first so the checkpoint exists in reports/checkpoints/.")
        st.stop()

    st.success(f"Loaded {model_name} on {device} from {ckpt_path}")

    img_size = int(cfg["data"].get("img_size", 224))

    tab_upload, tab_camera = st.tabs(["Upload image", "Camera input"])

    image: Image.Image | None = None

    with tab_upload:
        upload = st.file_uploader("Choose a leaf image", type=["png", "jpg", "jpeg", "webp"])
        if upload is not None:
            image = Image.open(upload).convert("RGB")

    with tab_camera:
        camera = st.camera_input("Capture a leaf image")
        if camera is not None and image is None:
            image = Image.open(camera).convert("RGB")

    if image is None:
        st.markdown(
            """
            <div class="card">
                <strong>How to use this page</strong><br/>
                1. Train the model with the dataset in data/raw/PlantVillage.<br/>
                2. Upload a leaf photo or capture one with your camera.<br/>
                3. Review the prediction and confidence below.
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    left, right = st.columns([1, 1])
    with left:
        st.image(image, caption="Input image", width="stretch")
    with right:
        result = predict_image(
            model=model,
            image=image,
            device=device,
            img_size=img_size,
            label_names=label_names,
            top_k=top_k,
        )
        render_prediction(result)

    st.markdown("---")
    st.markdown("#### Notes")
    st.write("The app loads the best checkpoint from reports/checkpoints/ by default. If you trained a different run, paste the checkpoint path in the sidebar.")


if __name__ == "__main__":
    main()