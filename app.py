
import streamlit as st
import numpy as np
import pywt
from scipy.sparse.linalg import svds
from PIL import Image
from supabase import create_client, Client
st.write("Streamlit version:", st.__version__)

st.set_page_config(
    page_title="Cloud Image Processing Service",
    page_icon="🖼️",
    layout="wide"
)
# -----------------------------
# Supabase connection
# -----------------------------

@st.cache_resource
def get_supabase_client():
    return create_client(
        st.secrets["SUPABASE_URL"],
        st.secrets["SUPABASE_ANON_KEY"]
    )

supabase = get_supabase_client()
#st.success("Supabase connection established successfully!")
st.write("Authentication status:", st.user.is_logged_in)

if st.user.is_logged_in:
    st.success("Google login successful!")
    st.write("User email:", st.user.get("email", "Email unavailable"))
    st.write("User name:", st.user.get("name", "Name unavailable"))
else:
    st.warning("User is not authenticated.")

    if st.button("Sign in with Google"):
        st.login()

    st.stop()

st.title("Cloud Image Processing as a Service")
st.write(
    "Gaussian noise, Wavelet denoising, "
    "and SVD-based image reconstruction."
)


# -----------------------------
# Quality metrics
# -----------------------------

def calculate_mse(original, processed):
    difference = original - processed
    return float(np.mean(difference ** 2))


def calculate_psnr(mse_value):
    if mse_value == 0:
        return float("inf")
    return 10 * np.log10(1.0 / mse_value)


# -----------------------------
# Soft thresholding
# -----------------------------

def soft_threshold(coefficients, threshold):
    return (
        np.sign(coefficients)
        * np.maximum(np.abs(coefficients) - threshold, 0)
    )


# -----------------------------
# Wavelet denoising
# -----------------------------

def wavelet_denoising(noisy_image, threshold_multiplier):

    coeffs = pywt.dwt2(noisy_image, wavelet="haar")

    approximation, (horizontal, vertical, diagonal) = coeffs

    sigma_estimated = (
        np.median(np.abs(diagonal)) / 0.6745
    )

    number_of_pixels = noisy_image.size

    threshold = (
        sigma_estimated
        * np.sqrt(2 * np.log(number_of_pixels))
        * threshold_multiplier
    )

    horizontal_filtered = soft_threshold(
        horizontal, threshold
    )

    vertical_filtered = soft_threshold(
        vertical, threshold
    )

    diagonal_filtered = soft_threshold(
        diagonal, threshold
    )

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

    denoised_image = denoised_image[
        :noisy_image.shape[0],
        :noisy_image.shape[1]
    ]

    denoised_image = np.clip(
        denoised_image, 0.0, 1.0
    )

    return denoised_image, sigma_estimated, threshold


# -----------------------------
# Truncated SVD reconstruction
# -----------------------------

def svd_reconstruction(image, rank):

    # Compute the largest rank singular components
    u, singular_values, vt = svds(
        image,
        k=rank,
        which="LM"
    )

    # svds returns singular values in ascending order.
    # Reverse them to get descending order.
    singular_values = singular_values[::-1]
    u = u[:, ::-1]
    vt = vt[::-1, :]

    # Reconstruct the image using the selected rank
    reconstructed = (
        u
        @ np.diag(singular_values)
        @ vt
    )

    reconstructed = np.clip(
        reconstructed, 0.0, 1.0
    )

    return reconstructed


# -----------------------------
# User interface
# -----------------------------

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["png", "jpg", "jpeg"]
)


if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("L")
    image.thumbnail((256, 256))

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

    max_rank = min(original.shape) - 1

    rank = st.sidebar.slider(
        "SVD rank (k)",
        min_value=1,
        max_value=max_rank,
        value=max(1, int(max_rank * 0.25)),
        step=1
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

    # Stage 1: Wavelet
    wavelet_image, estimated_sigma, threshold = (
        wavelet_denoising(
            noisy,
            threshold_multiplier
        )
    )

    # Stage 2: SVD applied to Wavelet output
    svd_image = svd_reconstruction(
        wavelet_image,
        rank
    )

    # -----------------------------
    # Display images
    # -----------------------------

    st.subheader("Image comparison")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.image(
            original,
            caption="Original",
            clamp=True,
            use_container_width=True
        )

    with col2:
        st.image(
            noisy,
            caption="Noisy",
            clamp=True,
            use_container_width=True
        )

    with col3:
        st.image(
            wavelet_image,
            caption="Wavelet denoising",
            clamp=True,
            use_container_width=True
        )

    with col4:
        st.image(
            svd_image,
            caption="Wavelet + SVD",
            clamp=True,
            use_container_width=True
        )

    # -----------------------------
    # Metrics
    # -----------------------------

    results = [
        ("Noisy", noisy),
        ("Wavelet", wavelet_image),
        ("Wavelet + SVD", svd_image)
    ]

    st.subheader("Quality measurements")

    metric_columns = st.columns(3)

    for column, (name, result_image) in zip(
        metric_columns, results
    ):

        mse_value = calculate_mse(
            original, result_image
        )

        psnr_value = calculate_psnr(mse_value)

        with column:
            st.markdown(f"**{name}**")
            st.metric("MSE", f"{mse_value:.6f}")

            if np.isinf(psnr_value):
                st.metric("PSNR", "Infinite")
            else:
                st.metric(
                    "PSNR",
                    f"{psnr_value:.2f} dB"
                )

    # -----------------------------
    # Parameters
    # -----------------------------

    st.subheader("Processing parameters")

    st.write("Wavelet: Haar (db1)")
    st.write("Wavelet decomposition level: 1")
    st.write(f"Estimated noise sigma: {estimated_sigma:.6f}")
    st.write(f"Wavelet threshold: {threshold:.6f}")
    st.write(f"SVD rank k: {rank}")
    st.write(f"Image dimensions: {original.shape}")

    st.caption(
        "SVD is applied to the Wavelet-denoised image. "
        "The selected rank k controls how many singular "
        "components are retained."
    )

else:
    st.info("Upload an image to begin.")
