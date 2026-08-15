from functools import partial
import os

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any
from dataclasses import dataclass

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

@dataclass
class ScatteringConfig:
    B: int =  32            # Batch size
    N: int = 88_200         # Time samples
    sr: float =  44_100.0   # Sampling rate
    beta: float = 3.0       # Morse beta parameter
    gamma: float =  20.0    # Morse gamma parameter
    J: int = 17             # Octaves
    Q1: int = 8             # Order 1 quality filters
    Q2: int = 1             # Order 2 quality filters
    ref_freq: float = sr/4  # Nyquist
    T: int = N              # Time-shift invariance count (single global one by default, if not this will give back floor(N/T) frames)

# Lowpass filter 
def gaussian_lowpass_filter(omega: jnp.ndarray, sigma: float) -> jnp.ndarray:
    return jnp.exp(-0.5 * (omega/sigma)**2)

# Bandpass
def bandpass_modulus(x: jnp.ndarray, psi_hat: jnp.ndarray) -> jnp.ndarray:
    x_hat = jnp.fft.fft(x, axis=1)
    W_hat = x_hat[:, None, :] * psi_hat[None, :, :]
    return jnp.abs(jnp.fft.ifft(W_hat, axis=-1))


def octave_frequencies(J: int, Q: int, ref_freq: float):
    j_idx = jnp.repeat(np.arange(J), Q)
    q_idx = jnp.tile(np.arange(Q), J)
    freqs =  (2.0 * ref_freq)**(-(j_idx + q_idx / Q))
    return freqs, j_idx

# Convolution filterbank building
def morse_filterbank(cfg: ScatteringConfig):
    # Buidling octaves and associated frequency range
    freqs_1, oct_1 = octave_frequencies(cfg.J, cfg.Q1, cfg.ref_freq)
    freqs_2, oct_2 = octave_frequencies(cfg.J, cfg.Q2, cfg.ref_freq)
    
    # Building scales from frequencies
    scales_1 = hz_to_scale(jnp.asarray(freqs_1), cfg.beta, cfg.gamma)
    scales_2 = hz_to_scale(jnp.asarray(freqs_2), cfg.beta, cfg.gamma)

    # Establish the frequency grids
    omega_full = 2.0*jnp.pi * jnp.fft.fftfreq(cfg.N, d=1.0/cfg.sr)
    omega_real = 2.0*jnp.pi * jnp.fft.rfftfreq(cfg.N, d=1.0/cfg.sr)

    # Making the Morse Wavelet fileterbanks
    morse = lambda s: morse_wavelet_freq_L1(s*omega_full, cfg.beta, cfg.gamma)
    psi_hat_1 = jax.vmap(morse)(scales_1).astype(jnp.float32)
    psi_hat_2 = jax.vmap(morse)(scales_2).astype(jnp.float32)

    # Lowpass filter
    sigma = jnp.pi * cfg.sr / cfg.T
    phi_hat = gaussian_lowpass_filter(omega_real, sigma).astype(jnp.float32)

    return [
        freqs_1, oct_1,
        freqs_2, oct_2,
        psi_hat_1, psi_hat_2,
        phi_hat
    ]

# Lowpass filter convolution
def lowpass(x: jnp.ndarray, phi_hat_real: jnp.ndarray, T: int) -> jnp.ndarray:
    N = x.shape[-1]
    X = jnp.fft.rfft(x, axis=-1) * phi_hat_real[None, None, :]
    return jnp.fft.irfft(X, n=N, axis=-1)[..., ::T]

# Wavelet scattering
def morse_scatter(cfg: ScatteringConfig=ScatteringConfig()):
    
    # Building filter bank first
    freqs_1, oct_1, freqs_2, oct_2, psi_hat_1, psi_hat_2, phi_hat = morse_filterbank(cfg)

    # Creating list of valid order-2 childeren per order 1 filter (octave(k2) > octave(k1))
    # Note the actual data list size is (most-likely) small therefore opted for building natively instead of parallelizing
    children: List[List[int]] = [
        [k2 for k2 in range(len(freqs_2)) if oct_2[k2] > oct_1[k1]]
        for k1 in range(len(freqs_1))
    ]

    # Extra metadata for the scatter
    P0: int = 1
    P1: int = cfg.J * cfg.Q1
    P2: int = cfg.J * cfg.Q1 * cfg.Q2 * (cfg.J - 1) // 2

    metadata: Dict[str, Any] = dict(
        J=cfg.J, Q1=cfg.Q1, Q2=cfg.Q2, T=cfg.T,
        P0=P0, P1=P1, P2=P2, dimension = P0 + P1 + P2,
        freqs1_hz=freqs_1, freqs2_hz=freqs_2, children=children,
    )


    # Scatter compute
    @jax.jit
    def scatter(signal: jnp.ndarray):
        # Convert signal to Float for consistency
        x = signal.astype(jnp.float32)

        # Order 0 scatter
        S0 = lowpass(x[:, None, :], phi_hat, cfg.T)

        # Order 1 scatter
        U1 = bandpass_modulus(x, psi_hat_1)
        S1 = lowpass(U1, phi_hat, cfg.T)

        # Order 2 scatter
        S2_buckets = [] # Wink-wink
        for k1 in range(P1):
            idx = children[k1]
            if not idx:
                continue
            idx_arr = jnp.asarray(idx)
            U2 =  bandpass_modulus(U1[:, k1, :], psi_hat_2[idx_arr])
            S2_buckets.append(lowpass(U2, phi_hat, cfg.T))
        
        S2 = (jnp.concatenate(S2_buckets, axis=1) if S2_buckets
              else jnp.zeros((signal.shape[0], 0, S0.shape[-1]), dtype=jnp.float32))

        # Return scatterings
        return S0, S1, S2

    return scatter, metadata
