import jax
import jax.numpy as jnp

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


def _tukey_window(length: int, taper_alpha: float=0.25) -> jnp.ndarray:
    """
    Tukey (tapered cosine) window
    """
    # Default to rectangular window setup for negative input taper_alpha
    if taper_alpha <= 0.0:
        return jnp.linspace(0.0, 1.0, length)
    
    # Tapers
    left = 0.5 * (1.0 - jnp.cos(2.0 * jnp.pi * x / alpha))
    right = 0.5 * (1.0 - jnp.cos(2.0 * jnp.pi * (1.0 - x) / alpha))
    
    # Piecewise selection
    return jnp.where(x < alpha / 2.0, left, jnp.where(x > 1.0 - alpha / 2.0, right, 1.0)).astype(jnp.float32)


def _morse_wavelet_freq_L1_single_scale(s: float, beta: float, gamma: float, length: int, taper: jnp.ndarray):
    """
    Vectorizing code for transform Morse wavelet to frequency domain across scales (L1 norm)
    """
    # Transforming 
    psi_hat = morse_wavelet_freq_L1(s*omega, beta, gamma)
    psi_t = jnp.fft.ifft(psi_hat)

    # Centering frequency and smoothing
    psi_t = jnp.fft.fftshift(psi_t)
    return psi_t * taper


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
        psi = morse_wavelet_freq_L1(s*omega, beta, gamma)
        return jnp.fft.fftfreq(x_hat * psi)
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