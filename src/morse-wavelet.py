from functools import partial
import os

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

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
#   Scattering implementation
###

def gaussian_low_pass_filter(omega: jnp.ndarray, sigma: float) -> jnp.ndarray:
    return jnp.exp(-0.5 * (omega/sigma)**2)

