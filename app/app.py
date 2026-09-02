"""
🌊 OceanTrace - Streamlit Web Application
AI-Powered Satellite Oil Spill Detection, Verification & Look-alike Discrimination System
"""

import os
import sys
import json
import tempfile
import urllib.request
import cv2
import numpy as np
import streamlit as st
import torch

# Ensure repository root is in python path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.preprocessing import apply_speckle_filter, normalize_image
from src.model import UNet
from src.verification import verify_oil_spill_detection
from src.tracking import simulate_slick_drift, draw_drift_trajectory

# ── Cloud deployment: auto-download model weights from GitHub Releases ──────
GITHUB_RELEASE_BASE = (
    "https://github.com/chanderbala61-jpg/Oceantrace/releases/download/v1.0.0"
)
MODEL_FILES = {
    "best_unet_model_fast.pth": f"{GITHUB_RELEASE_BASE}/best_unet_model_fast.pth",
    "best_unet_model.pth": f"{GITHUB_RELEASE_BASE}/best_unet_model.pth",
}


def ensure_model_downloaded():
    """Download model checkpoints from GitHub Releases if not present locally."""
    checkpoint_dir = os.path.join(ROOT_DIR, "models", "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    for filename, url in MODEL_FILES.items():
        dest = os.path.join(checkpoint_dir, filename)
        if not os.path.exists(dest):
            try:
                st.info(f"⬇️ Downloading model weights: {filename} …")
                urllib.request.urlretrieve(url, dest)
                st.success(f"✅ Downloaded {filename}")
            except Exception as e:
                st.warning(f"⚠️ Could not download {filename}: {e}")


ensure_model_downloaded()


# Streamlit Page Configuration
st.set_page_config(
    page_title="OceanTrace - AI Satellite Oil Spill Detection",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Dark Cyber-Ocean Theme)
st.markdown("""
    <style>
    .main {
        background-color: #0b132b;
        color: #e0e1dd;
    }
    .stButton>button {
        background-color: #00b4d8;
        color: #ffffff;
        font-weight: bold;
        border-radius: 8px;
        border: none;
        padding: 10px 24px;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #90e0ef;
        color: #0b132b;
    }
    .metric-box {
        background-color: #1c2541;
        padding: 16px;
        border-radius: 10px;
        border: 1px solid #3a506b;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_trained_model(checkpoint_path: str):
    """Loads and caches PyTorch U-Net or UNetFast checkpoint."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Fallback to best_unet_model_fast.pth if default does not exist
    if not os.path.exists(checkpoint_path):
        alt_path = os.path.join(os.path.dirname(checkpoint_path), 'best_unet_model_fast.pth')
        if os.path.exists(alt_path):
            checkpoint_path = alt_path
        else:
            # Last resort: try downloading on-demand
            ensure_model_downloaded()
            if os.path.exists(alt_path):
                checkpoint_path = alt_path

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if checkpoint.get('architecture') == 'UNetFast' or checkpoint.get('base_channels') == 16:
            from src.train_fast import UNetFast
            model = UNetFast(in_channels=3, out_channels=1, base_channels=checkpoint.get('base_channels', 16)).to(device)
        else:
            model = UNet(in_channels=3, out_channels=1).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Load associated metrics if available
        metrics = checkpoint.get('metrics', {})
        val_dice = metrics.get('val_dice', None)
        val_iou = metrics.get('val_iou', None)
        
        # Check train history json
        hist_path = os.path.join(os.path.dirname(checkpoint_path), 'train_history_fast.json')
        if os.path.exists(hist_path):
            try:
                with open(hist_path, 'r') as hf:
                    hist = json.load(hf)
                    if hist.get('val_dice'):
                        val_dice = max(hist['val_dice'])
                    if hist.get('val_iou'):
                        val_iou = max(hist['val_iou'])
            except Exception:
                pass
                
        return model, device, True, os.path.basename(checkpoint_path), val_dice, val_iou
    else:
        model = UNet(in_channels=3, out_channels=1).to(device)
        model.eval()
        return model, device, False, "None", None, None


def predict_sliding_window(
    image_rgb: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
    patch_size: int = 256,
    stride: int = 192
) -> np.ndarray:
    """
    Executes high-resolution sliding window tiled inference to prevent texture destruction
    from whole-scene downsampling.
    """
    H, W = image_rgb.shape[:2]
    
    # Fast path for smaller images
    if max(H, W) <= 512:
        input_resized = cv2.resize(image_rgb, (patch_size, patch_size), interpolation=cv2.INTER_LINEAR)
        input_tensor = torch.from_numpy(input_resized.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
        with torch.no_grad():
            logits = model(input_tensor)
            probs = torch.sigmoid(logits)[0, 0].cpu().numpy()
        return cv2.resize(probs, (W, H), interpolation=cv2.INTER_LINEAR)

    # Sliding window inference
    prob_accumulator = np.zeros((H, W), dtype=np.float32)
    weight_accumulator = np.zeros((H, W), dtype=np.float32)

    # 2D Gaussian weight window for smooth tile blending
    y_win = np.hanning(patch_size)
    x_win = np.hanning(patch_size)
    window_2d = np.outer(y_win, x_win) + 1e-3

    y_steps = range(0, max(1, H - patch_size + 1), stride)
    x_steps = range(0, max(1, W - patch_size + 1), stride)

    # Ensure last edges are covered
    y_coords = list(y_steps)
    if y_coords[-1] + patch_size < H:
        y_coords.append(H - patch_size)
        
    x_coords = list(x_steps)
    if x_coords[-1] + patch_size < W:
        x_coords.append(W - patch_size)

    for y in y_coords:
        for x in x_coords:
            patch = image_rgb[y:y+patch_size, x:x+patch_size]
            patch_tensor = torch.from_numpy(patch.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
            with torch.no_grad():
                logits = model(patch_tensor)
                patch_prob = torch.sigmoid(logits)[0, 0].cpu().numpy()

            prob_accumulator[y:y+patch_size, x:x+patch_size] += patch_prob * window_2d
            weight_accumulator[y:y+patch_size, x:x+patch_size] += window_2d

    final_prob = prob_accumulator / np.maximum(weight_accumulator, 1e-3)
    return final_prob


# Header & Branding
st.title("🌊 OceanTrace")
st.caption("AI-Powered Satellite Oil Spill Detection, Verification & Look-alike Discrimination System")
st.markdown("---")

# Sidebar Controls
st.sidebar.header("⚙️ Model Controls & Benchmark Accuracy")
checkpoint_path = os.path.join(ROOT_DIR, 'models', 'checkpoints', 'best_unet_model_fast.pth')
model, device, is_model_loaded, ckpt_name, val_dice, val_iou = load_trained_model(checkpoint_path)

if is_model_loaded:
    st.sidebar.success(f"✅ Active Model: `{ckpt_name}`")
    
    # Display Benchmark Accuracy / Validation Metrics
    st.sidebar.markdown("### 📈 Model Evaluation Metrics")
    col_sb1, col_sb2 = st.sidebar.columns(2)
    with col_sb1:
        st.metric("Validation IoU", f"{val_iou:.1%}" if val_iou is not None else "N/A")
    with col_sb2:
        st.metric("Dice Score (F1)", f"{val_dice:.1%}" if val_dice is not None else "N/A")
    st.sidebar.caption("Evaluated on independent validation dataset patches.")
else:
    st.sidebar.warning("⚠️ Model Checkpoint Not Found. Using Initialized Model.")

st.sidebar.markdown(f"**Execution Device:** `{device.type.upper()}`")

st.sidebar.subheader("Detection Parameters")
threshold = st.sidebar.slider("Prediction Probability Threshold", min_value=0.1, max_value=0.9, value=0.5, step=0.05)
filter_method = st.sidebar.selectbox("SAR Speckle Noise Reduction Filter", ["median", "lee", "none"])
inference_mode = st.sidebar.radio("Inference Strategy", ["High-Res Sliding Window (Accurate)", "Whole Scene Fast Resize"], index=0)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚠️ Decision Support Tool")
st.sidebar.info("OceanTrace employs physics-based SAR look-alike discrimination (damping ratio & border gradient) to filter low-wind zones and biogenic slicks.")


# Main Application Interface
uploaded_file = st.file_uploader("Upload Satellite Image (SAR / TIFF / PNG / JPG)", type=["tif", "tiff", "png", "jpg", "jpeg"])

# Quick Sample Image Selector if no file uploaded
if uploaded_file is None:
    st.info("💡 No file uploaded yet. You can upload your own satellite scene or test with a sample scene from the dataset below:")
    sample_dir = os.path.join(ROOT_DIR, 'data', 'raw', 'train', 'images')
    if os.path.exists(sample_dir):
        sample_files = [f for f in os.listdir(sample_dir) if f.lower().endswith('.tif')]
        if sample_files:
            selected_sample = st.selectbox("Select Sample Satellite Scene", sample_files)
            if st.button("Load Selected Sample Scene"):
                sample_path = os.path.join(sample_dir, selected_sample)
                uploaded_file = sample_path


if uploaded_file is not None:
    st.markdown("### 🔍 Image Analysis & AI Inference")
    
    # Read Image File
    try:
        from PIL import Image
        import io
        if isinstance(uploaded_file, str):
            with Image.open(uploaded_file) as pil_img:
                image_rgb = np.array(pil_img)
        else:
            with Image.open(io.BytesIO(uploaded_file.read())) as pil_img:
                image_rgb = np.array(pil_img)
        if image_rgb.ndim == 2:
            image_rgb = np.stack([image_rgb, image_rgb, image_rgb], axis=-1)
        elif image_rgb.shape[-1] == 4:
            image_rgb = image_rgb[:, :, :3]
            
        # Ensure image_rgb and image_bgr are strictly uint8 in [0, 255]
        if image_rgb.dtype != np.uint8:
            min_val = float(np.min(image_rgb))
            max_val = float(np.max(image_rgb))
            if max_val > min_val:
                image_rgb = np.clip((image_rgb - min_val) / (max_val - min_val) * 255.0, 0, 255).astype(np.uint8)
            else:
                image_rgb = np.zeros_like(image_rgb, dtype=np.uint8)
                
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    except Exception:
        if isinstance(uploaded_file, str):
            image_bgr = cv2.imread(uploaded_file, cv2.IMREAD_COLOR)
        else:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if image_bgr is not None:
            if image_bgr.dtype != np.uint8:
                image_bgr = np.clip(image_bgr, 0, 255).astype(np.uint8)
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = None
        
    if image_rgb is None or image_bgr is None:
        st.error("Error decoding satellite image file. Please upload a valid image.")
    else:
        H, W = image_rgb.shape[:2]
        
        col_btn1, col_btn2 = st.columns([1, 4])
        with col_btn1:
            run_predict = st.button("🚀 RUN DETECTION")
            
        if run_predict or 'pred_results' in st.session_state:
            with st.spinner("Processing SAR Image & Running Neural Inference..."):
                # Preprocessing
                filtered_img = apply_speckle_filter(image_rgb, method=filter_method, kernel_size=3)
                norm_img = normalize_image(filtered_img, method='minmax')
                
                # Inference execution
                if "Sliding Window" in inference_mode:
                    prob_map = predict_sliding_window(norm_img, model, device, patch_size=256, stride=192)
                else:
                    input_resized = cv2.resize(norm_img, (256, 256), interpolation=cv2.INTER_LINEAR)
                    input_tensor = torch.from_numpy(input_resized.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
                    with torch.no_grad():
                        logits = model(input_tensor)
                        probs_resized = torch.sigmoid(logits)[0, 0].cpu().numpy()
                    prob_map = cv2.resize(probs_resized, (W, H), interpolation=cv2.INTER_LINEAR)
                    
                binary_mask = (prob_map >= threshold).astype(np.uint8)
                
                # Verification & Look-alike Analysis
                verif_results = verify_oil_spill_detection(
                    prob_map,
                    binary_mask,
                    original_image=image_rgb,
                    threshold=threshold
                )
                
                st.session_state['pred_results'] = {
                    'image_rgb': image_rgb,
                    'image_bgr': image_bgr,
                    'prob_map': prob_map,
                    'binary_mask': binary_mask,
                    'verif': verif_results
                }
            
            st.markdown("---")
            st.subheader("📊 RESULT SUMMARY & ACCURACY / CONFIDENCE METRICS")
            
            # Status Banner & Classification Badge
            classification = verif_results.get('classification', 'CONFIRMED OIL SPILL')
            if classification == "CONFIRMED OIL SPILL":
                st.error(f"🚨 **{classification}** (Model Confidence: **{verif_results['confidence']}** | Risk: **{verif_results['risk_level']}**)")
            elif classification == "SUSPECTED LOOK-ALIKE":
                st.warning(f"⚠️ **{classification}** (Model Confidence: **{verif_results['confidence']}** | Risk: **{verif_results['risk_level']}**)")
            elif classification == "POTENTIAL SPILL / INVESTIGATE":
                st.info(f"🔎 **{classification}** (Model Confidence: **{verif_results['confidence']}**)")
            else:
                st.success("✅ **CLEAN MARINE REGION** (No Oil Spill Detected)")
                
            # Key Metrics Cards including Detection Certainty & Model Accuracy
            m1, m2, m3, m4, m5 = st.columns(5)
            with m1:
                st.metric("Classification", verif_results.get('classification', 'N/A'))
            with m2:
                st.metric("Mean Detection Certainty", f"{verif_results['mean_prob']:.1%}")
            with m3:
                st.metric("Peak Probability", f"{verif_results['max_prob']:.1%}")
            with m4:
                st.metric("Damping Contrast", f"{verif_results.get('damping_ratio', 1.0):.2f}x")
            with m5:
                st.metric("Detected Pixels", f"{verif_results['spill_pixels']:,}")
                
            st.info(f"**Detailed SAR Decision Explanation:** {verif_results['explanation']}")
            
            # Image & Prediction Display Tabs
            tab1, tab2, tab3 = st.columns([1, 1, 1])
            
            # Create Color Overlay (Red highlight)
            red_mask = np.zeros_like(image_bgr)
            red_mask[:, :, 2] = binary_mask * 255
            overlay_bgr = cv2.addWeighted(image_bgr, 0.7, red_mask, 0.5, 0)
            overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
            
            with tab1:
                st.markdown("##### Original Satellite Image")
                st.image(image_rgb, use_container_width=True, clamp=True)
                
            with tab2:
                st.markdown("##### AI Binary Mask")
                st.image(binary_mask * 255, use_container_width=True, clamp=True)
                
            with tab3:
                st.markdown("##### Color Overlay")
                st.image(overlay_rgb, use_container_width=True, clamp=True)
                
            st.markdown("---")
            
            # Drift Tracking Simulation Sub-Section
            st.subheader("🛥️ Prototype Drift & Trajectory Tracking (Simulation)")
            
            if verif_results['spill_pixels'] > 0 and not verif_results.get('is_lookalike', False):
                t1, t2 = st.columns([1, 2])
                with t1:
                    wind_spd = st.slider("Wind Speed (Knots)", 0.0, 30.0, 12.0)
                    wind_dir = st.slider("Wind Direction (Degrees)", 0, 360, 45)
                    fcst_hrs = st.slider("Forecast Horizon (Hours)", 1, 24, 6)
                    
                    drift_info = simulate_slick_drift(
                        binary_mask,
                        wind_speed_knots=wind_spd,
                        wind_direction_deg=wind_dir,
                        forecast_hours=fcst_hrs
                    )
                    
                    st.write(f"**Projected Drift Distance:** `{drift_info['drift_distance_nm']} NM`")
                    st.write(f"**Drift Heading:** `{drift_info['drift_heading_deg']}°`")
                    
                with t2:
                    drift_vis_rgb = draw_drift_trajectory(image_rgb, binary_mask, drift_info)
                    st.image(drift_vis_rgb, caption="Prototype Drift Trajectory Vector Simulation", use_container_width=True)
            elif verif_results.get('is_lookalike', False):
                st.warning("Drift simulation bypassed: The detected anomaly is classified as a probable Look-alike (low-wind/biogenic film).")
            else:
                st.info("Drift tracking simulation is inactive because no oil slick region was detected.")
