"""
🌊 OceanTrace - Streamlit Web Application
AI-Powered Satellite Oil Spill Detection, Verification & Tracking System
"""

import os
import sys
import tempfile
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

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if checkpoint.get('architecture') == 'UNetFast' or checkpoint.get('base_channels') == 16:
            from src.train_fast import UNetFast
            model = UNetFast(in_channels=3, out_channels=1, base_channels=checkpoint.get('base_channels', 16)).to(device)
        else:
            model = UNet(in_channels=3, out_channels=1).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model, device, True, os.path.basename(checkpoint_path)
    else:
        model = UNet(in_channels=3, out_channels=1).to(device)
        model.eval()
        return model, device, False, "None"


# Header & Branding
st.title("🌊 OceanTrace")
st.caption("AI-Powered Satellite Oil Spill Detection, Verification & Tracking System")
st.markdown("---")

# Sidebar Controls
st.sidebar.header("⚙️ Model Controls")
checkpoint_path = os.path.join(ROOT_DIR, 'models', 'checkpoints', 'best_unet_model_fast.pth')
model, device, is_model_loaded, ckpt_name = load_trained_model(checkpoint_path)

if is_model_loaded:
    st.sidebar.success(f"✅ Model Checkpoint Loaded (`{ckpt_name}`)")
else:
    st.sidebar.warning("⚠️ Model Checkpoint Not Found. Using Initialized Model.")

st.sidebar.markdown(f"**Execution Device:** `{device.type.upper()}`")

st.sidebar.subheader("Detection Parameters")
threshold = st.sidebar.slider("Prediction Probability Threshold", min_value=0.1, max_value=0.9, value=0.5, step=0.05)
filter_method = st.sidebar.selectbox("SAR Speckle Noise Reduction Filter", ["median", "lee", "none"])

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚠️ Decision Support Tool")
st.sidebar.info("OceanTrace is an AI decision-support prototype. Detection results require validation by maritime authorities.")


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
                with open(sample_path, "rb") as f:
                    uploaded_bytes = f.read()
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
            # Preprocessing & Forward Pass
            filtered_img = apply_speckle_filter(image_rgb, method=filter_method, kernel_size=3)
            norm_img = normalize_image(filtered_img, method='minmax')
            
            input_resized = cv2.resize(norm_img, (256, 256), interpolation=cv2.INTER_LINEAR)
            input_tensor = torch.from_numpy(input_resized.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
            
            with torch.no_grad():
                logits = model(input_tensor)
                probs_resized = torch.sigmoid(logits)[0, 0].cpu().numpy()
                
            prob_map = cv2.resize(probs_resized, (W, H), interpolation=cv2.INTER_LINEAR)
            binary_mask = (prob_map >= threshold).astype(np.uint8)
            
            # Verification Analysis
            verif_results = verify_oil_spill_detection(prob_map, binary_mask, threshold=threshold)
            
            st.session_state['pred_results'] = {
                'image_rgb': image_rgb,
                'image_bgr': image_bgr,
                'prob_map': prob_map,
                'binary_mask': binary_mask,
                'verif': verif_results
            }
            
            st.markdown("---")
            st.subheader("📊 RESULT SUMMARY")
            
            # Status Banner
            if verif_results['spill_pixels'] > 0:
                st.error(f"🛢️ **POTENTIAL OIL SPILL DETECTED** (Region Confidence: **{verif_results['confidence']}**)")
            else:
                st.success("✅ **NO OIL SPILL DETECTED** (Clean Marine Region)")
                
            # Key Metrics Cards
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Prototype Confidence", verif_results['confidence'])
            with m2:
                st.metric("Risk Level", verif_results['risk_level'])
            with m3:
                st.metric("Detected Spill Pixels", f"{verif_results['spill_pixels']:,}")
            with m4:
                st.metric("Region Max Probability", f"{verif_results['max_prob']:.2%}")
                
            st.info(f"**Explanation:** {verif_results['explanation']}")
            
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
            
            if verif_results['spill_pixels'] > 0:
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
            else:
                st.info("Drift tracking simulation is inactive because no oil slick region was detected.")
