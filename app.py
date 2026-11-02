
import streamlit as st
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Cloud Image Processing Service",
    page_icon="🖼️",
    layout="wide"
)

st.title("Cloud Image Processing as a Service")
st.write(
    "Upload an image, add Gaussian noise, "
    "and evaluate image quality."
)

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["png", "jpg", "jpeg"]
)

def calculate_mse(original, processed):
    difference = original - processed
    return float(np.mean(difference ** 2))

def calculate_psnr(mse_value):
    if mse_value == 0:
        return float("inf")
    return 10 * np.log10(1.0 / mse_value)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")
    image.thumbnail((512, 512))

    original = np.asarray(
        image, dtype=np.float64
    ) / 255.0

    st.sidebar.header("Processing settings")

    noise_level = st.sidebar.slider(
        "Gaussian noise standard deviation",
        min_value=0.0,
        max_value=0.20,
        value=0.05,
        step=0.01
    )

    rng = np.random.default_rng(42)

    noise = rng.normal(
        loc=0.0,
        scale=noise_level,
        size=original.shape
    )

    noisy = np.clip(original + noise, 0.0, 1.0)

    st.subheader("Image comparison")

    col1, col2 = st.columns(2)

    with col1:
        st.image(
            original,
            caption="Original grayscale image",
            clamp=True,
            use_container_width=True
        )

    with col2:
        st.image(
            noisy,
            caption="Image with Gaussian noise",
            clamp=True,
            use_container_width=True
        )

    mse_value = calculate_mse(original, noisy)
    psnr_value = calculate_psnr(mse_value)

    st.subheader("Quality measurements")

    metric1, metric2 = st.columns(2)

    metric1.metric("MSE", f"{mse_value:.6f}")

    if np.isinf(psnr_value):
        metric2.metric("PSNR", "Infinite")
    else:
        metric2.metric(
            "PSNR",
            f"{psnr_value:.2f} dB"
        )

    st.caption(
        "MSE and PSNR compare the noisy image with "
        "the original reference image. Lower MSE "
        "and higher PSNR generally indicate less distortion."
    )

else:
    st.info("Upload an image to begin.")
