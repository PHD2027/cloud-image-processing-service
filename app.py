
import streamlit as st
import numpy as np
import pywt
from PIL import Image

st.set_page_config(
    page_title="Cloud Image Processing Service",
    page_icon="🖼️",
    layout="wide"
)

st.title("Cloud Image Processing as a Service")
st.write(
    "Upload an image, add Gaussian noise, "
    "and compare Wavelet denoising results."
)


def calculate_mse(original, processed):
    difference = original - processed
    return float(np.mean(difference ** 2))


def calculate_psnr(mse_value):
    if mse_value == 0:
        return float("inf")
    return 10 * np.log10(1.0 / mse_value)


def soft_threshold(coefficients, threshold):
    """
    Soft thresholding:
    sign(c) * max(|c| - threshold, 0)
    """
    return (
        np.sign(coefficients)
        * np.maximum(np.abs(coefficients) - threshold, 0)
    )


def wavelet_denoising(noisy_image, threshold_multiplier):
    """
    One-level 2D Haar wavelet denoising.
    Approximation coefficients are preserved.
    Detail coefficients are soft-thresholded.
    """

    # Step 1: Apply 2D Discrete Wavelet Transform
    coeffs = pywt.dwt2(noisy_image, wavelet="haar")

    approximation, (horizontal, vertical, diagonal) = coeffs

    # Step 2: Estimate noise standard deviation
    sigma_estimated = (
        np.median(np.abs(diagonal)) / 0.6745
    )

    # Step 3: Calculate universal threshold
    number_of_pixels = noisy_image.size

    threshold = (
        sigma_estimated
        * np.sqrt(2 * np.log(number_of_pixels))
        * threshold_multiplier
    )

    # Step 4: Apply soft thresholding to detail coefficients
    horizontal_filtered = soft_threshold(
        horizontal, threshold
    )

    vertical_filtered = soft_threshold(
        vertical, threshold
    )

    diagonal_filtered = soft_threshold(
        diagonal, threshold
    )

    # Step 5: Reconstruct image using inverse DWT
    filtered_coeffs = (
        approximation,
        (
            horizontal_filtered,
            vertical_filtered,
            diagonal_filtered
        )
    )

    denoised_image = pywt.idwt2(
        filtered_coeffs,
        wavelet="haar"
    )

    # Match reconstructed dimensions to input dimensions
    denoised_image = denoised_image[
        :noisy_image.shape[0],
        :noisy_image.shape[1]
    ]

    denoised_image = np.clip(
        denoised_image, 0.0, 1.0
    )

    return denoised_image, sigma_estimated, threshold


uploaded_file = st.file_uploader(
    "Choose an image",
    type=["png", "jpg", "jpeg"]
)


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

    threshold_multiplier = st.sidebar.slider(
        "Wavelet threshold multiplier",
        min_value=0.5,
        max_value=2.0,
        value=1.0,
        step=0.1
    )

    rng = np.random.default_rng(42)

    noise = rng.normal(
        loc=0.0,
        scale=noise_level,
        size=original.shape
    )

    noisy = np.clip(
        original + noise,
        0.0,
        1.0
    )

    # Apply Wavelet denoising
    denoised, estimated_sigma, threshold = (
        wavelet_denoising(
            noisy,
            threshold_multiplier
        )
    )

    st.subheader("Image comparison")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(
            original,
            caption="Original image",
            clamp=True,
            use_container_width=True
        )

    with col2:
        st.image(
            noisy,
            caption="Noisy image",
            clamp=True,
            use_container_width=True
        )

    with col3:
        st.image(
            denoised,
            caption="Wavelet-denoised image",
            clamp=True,
            use_container_width=True
        )

    # Calculate quality metrics
    mse_noisy = calculate_mse(original, noisy)
    psnr_noisy = calculate_psnr(mse_noisy)

    mse_denoised = calculate_mse(original, denoised)
    psnr_denoised = calculate_psnr(mse_denoised)

    st.subheader("Quality measurements")

    metric1, metric2 = st.columns(2)

    with metric1:
        st.markdown("**Noisy image**")
        st.metric("MSE", f"{mse_noisy:.6f}")

        if np.isinf(psnr_noisy):
            st.metric("PSNR", "Infinite")
        else:
            st.metric("PSNR", f"{psnr_noisy:.2f} dB")

    with metric2:
        st.markdown("**Wavelet-denoised image**")
        st.metric("MSE", f"{mse_denoised:.6f}")

        if np.isinf(psnr_denoised):
            st.metric("PSNR", "Infinite")
        else:
            st.metric("PSNR", f"{psnr_denoised:.2f} dB")

    st.subheader("Wavelet parameters")

    st.write(f"Wavelet: Haar (db1)")
    st.write("Decomposition level: 1")
    st.write(f"Estimated noise sigma: {estimated_sigma:.6f}")
    st.write(f"Applied threshold: {threshold:.6f}")

    st.caption(
        "Wavelet denoising preserves approximation coefficients "
        "and applies soft thresholding to detail coefficients. "
        "MSE and PSNR are calculated against the original image."
    )

else:
    st.info("Upload an image to begin.")
