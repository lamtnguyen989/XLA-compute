from functools import partial
import os

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

DATA_DIR = os.path.join(os.path.dirname(__file__), "../data") if "__file__" in locals() else "./data"
OUT_DIR = os.path.join(os.path.dirname(__file__), "../output") if "__file__" in locals() else "./output"
os.makedirs(OUT_DIR, exist_ok=True)

###
# Morse Wavelet computation
###

def morse_log_L1_normalization(beta: float, gamma: float) -> jnp.ndarray:
    """
    Calculate the log of bandwidth (L1) normalization constant to the Morse Wavelet
    """
    return jnp.log(2.0) + (beta/gamma)*(1.0 + jnp.log(gamma) - jnp.log(beta))


def morse_peak_frequency(beta: float, gamma: float) -> jnp.ndarray:
    """
    Peak Morse Wavelet frequency
    """
    return (beta/gamma)**(1.0/gamma)

def morse_wavelet_freq_L1(omega: jnp.ndarray, beta: float, gamma: float) -> jnp.ndarray:
    """
    Morse Wavelet in frequency domain under L1 normalization.
    """
    log_norm_const = morse_log_L1_normalization(beta, gamma)
    analytic_omega = jnp.where(omega > 0, omega, 1.0)
    log_mag = log_norm_const + beta * jnp.log(analytic_omega) - analytic_omega**gamma
    mag = jnp.exp(log_mag)
    return jnp.where(omega > 0, mag, 0.0)


# def _tukey_window(length: int, taper_alpha: float=0.25) -> jnp.ndarray:
#     """
#     Tukey (tapered cosine) window
#     """
#     # Default to rectangular window setup for negative input taper_alpha
#     if taper_alpha <= 0.0:
#         return jnp.linspace(0.0, 1.0, length)
    
#     # Tapers
#     left = 0.5 * (1.0 - jnp.cos(2.0 * jnp.pi * x / alpha))
#     right = 0.5 * (1.0 - jnp.cos(2.0 * jnp.pi * (1.0 - x) / alpha))
    
#     # Piecewise selection
#     return jnp.where(x < alpha / 2.0, left, jnp.where(x > 1.0 - alpha / 2.0, right, 1.0)).astype(jnp.float32)


# def _morse_wavelet_freq_L1_single_scale(s: float, beta: float, gamma: float, length: int, taper: jnp.ndarray):
#     """
#     Vectorizing code for transform Morse wavelet to frequency domain across scales (L1 norm)
#     """
#     # Transforming 
#     psi_hat = morse_wavelet_freq_L1(s*omega, beta, gamma)
#     psi_t = jnp.fft.ifft(psi_hat)

#     # Centering frequency and smoothing
#     psi_t = jnp.fft.fftshift(psi_t)
#     return psi_t * taper


# def morse_wavelet_L1_kernels(scales: jnp.ndarray, beta: float, gamma: float, fs:float, length: int, taper_alpha: float = 0.25):
#     """
#     Pre-compute FIR kernels for the Morse Wavelets
#     """
#     # Setting up domain and taper
#     omega =  (2.0*jnp.pi) * jnp.fft.fftfreq(kernel_length, d=1.0 / fs)
#     taper = _tukey_window(beta, gamma, taper_alpha)

#     # Create kernels across scales
#     def _single_scale_kernel(s):
#         """
#         Single scale internel code for vectorizing transforms across scales
#         """
#         # Transforming 
#         psi_hat = morse_wavelet_freq_L1(s*omega, beta, gamma)
#         psi_t = jnp.fft.ifft(psi_hat)

#         # Centering frequency and smoothing
#         psi_t = jnp.fft.fftshift(psi_t)
#         return psi_t * taper

#     return jax.vmap(_single_scale_kernel)(scales)



@partial(jax.jit, static_argnames=())
def morse_transform_L1(signal: jnp.ndarray, scales: jnp.ndarray, 
                    beta: float = 3.0, gamma: float = 60.0, fs: float = 1.0) -> jnp.ndarray:
    """
    Continuous Wavelet Transformation of the signal based on Morse mother Wavelet
    """
    # Frequency domain setup
    n = signal.shape[-1]
    x_hat = jnp.fft.fft(signal)
    omega = (2.0*jnp.pi) * jnp.fft.fftfreq(n, d=1.0/fs)

    # Transform 
    def _transform_at_scale(s: jnp.float32):
        psi = morse_wavelet_freq_L1(s * omega, beta, gamma)
        return jnp.fft.ifft(x_hat * psi)

    return jax.vmap(_transform_at_scale)(scales)


###
# Domain helper
###

def scale_to_hz(scale: jnp.ndarray, beta: float, gamma: float) -> jnp.ndarray:
    omega_peak = morse_peak_frequency(beta, gamma)
    return omega_peak / (2.0 * jnp.pi * scale)


def hz_to_scale(freq_hz: jnp.ndarray, beta: float, gamma: float) -> jnp.ndarray:
    omega_peak = morse_peak_frequency(beta, gamma)
    return omega_peak / (2.0 * jnp.pi * freq_hz)


def log_scales(n_scales: int, f_min: float, f_max: float, beta: float = 3.0, gamma: float = 60.0,) -> jnp.ndarray:
    freqs = jnp.geomspace(f_min, f_max, n_scales)
    return hz_to_scale(freqs, beta, gamma)


###
# Scalogram
###

def scalogram(
    signal: np.ndarray,
    sr: int,
    f_min: float = 20.0,
    f_max: float = 15000.0,
    n_scales: int = 64,
    beta: float = 3.0,
    gamma: float = 60.0,
    title: str = "Audio Scalogram",
    out_path: str = None,
):
    scales = log_scales(n_scales, f_min, f_max, beta=beta, gamma=gamma)
    freqs = np.asarray(scale_to_hz(scales, beta, gamma))

    W = morse_transform_L1(
        jnp.asarray(signal),
        scales.astype(jnp.float32),
        beta=beta,
        gamma=gamma,
        fs=float(sr),
    )

    mag = np.asarray(jnp.abs(W))
    t = np.arange(signal.shape[0]) / float(sr)

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.pcolormesh(t, freqs, mag, shading="auto", cmap="magma")
    ax.set_yscale("log")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="|W|")
    fig.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()


###
# Audio test Runner
###

def read_wav(filename: str) -> tuple[np.ndarray, int]:
    path = os.path.join(DATA_DIR, filename)
    signal, sr = sf.read(path, dtype="float32", always_2d=False)
    if signal.ndim > 1:
        signal = signal.mean(axis=1).astype(np.float32)
    return signal, sr


def run_case(
    filename: str,
    title: str = "Audio Scalogram",
    f_low: float = 20.0,
    f_high: float = 15000.0,
    max_seconds: float = 10.0,
    beta: float = 3.0,
    gamma: float = 60.0,
):
    signal, sr = read_wav(filename)

    max_samples = int(max_seconds * sr)
    if signal.shape[0] > max_samples:
        signal = signal[:max_samples]

    out_path = os.path.join(OUT_DIR, filename.replace(".wav", "_scalogram.png"))

    scalogram(
        signal,
        sr,
        f_min=max(f_low, 1.0),
        f_max=f_high,
        beta=beta,
        gamma=gamma,
        title=title,
        out_path=out_path,
    )


if __name__ == "__main__":
    run_case("exp_chirp_20_10000hz.wav")
    run_case("linear_chirp_20_10000hz.wav")