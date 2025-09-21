# -*- coding: utf-8 -*-
"""
Created on Sun Sep 21 10:43:45 2025

@author: Bertrand
"""
import numpy as np


def make_bpsk_tx_buffer(sample_rate, fc_hz, pri_samples, chirp_start_samples,
                        chirp_time_s, Lc=16, seed=123):
    """
    Build one-PRI complex TX buffer: a tone at fc_hz multiplied by a ±1 chip sequence.
    The chips occupy exactly the chirp_time_s window starting at chirp_start_samples.

    Returns:
      tx_iq: complex64 array length pri_samples
      code_bits: the ±1 chips used (length Lc)
    """
    N = int(pri_samples)
    n = np.arange(N, dtype=np.float64)

    # Baseband tone (continuous over the full PRI)
    phase = 2.0 * np.pi * fc_hz * n / float(sample_rate)
    tone = np.exp(1j * phase).astype(np.complex64)

    # Where in the PRI the chirp starts and how long it lasts
    s0 = int(chirp_start_samples)
    Lr = int(round(chirp_time_s * sample_rate))

    # Build Lc chips that fill Lr samples (last chip stretched/shrunk to fit)
    rng = np.random.default_rng(seed)
    code_bits = rng.choice([1.0, -1.0], size=Lc).astype(np.float32)

    spc = max(1, Lr // Lc)                # nominal samples per chip
    chips = np.repeat(code_bits, spc)     # length ~ Lr, then trim/pad
    if chips.size < Lr:
        # pad by repeating last chip
        chips = np.pad(chips, (0, Lr - chips.size), mode="edge")
    elif chips.size > Lr:
        chips = chips[:Lr]

    # Full-PRI code vector (1s outside chirp window)
    code_vec = np.ones(N, dtype=np.float32)
    e = min(N, s0 + Lr)
    if e > s0:                            # write chips into the chirp window
        code_vec[s0:e] = chips[:e - s0]

    # Apply BPSK: multiply the tone by ±1, scale to Pluto range
    tx_iq = (tone * code_vec.astype(np.complex64)) * (2 ** 14)
    return tx_iq.astype(np.complex64), code_bits




def design_pc_allpass(fs, B, T, Nfft, fb_max=None):
    """
    Make an all-pass H(f) with group delay tau_GD(fb) = (1/k)*(fb_max - fb),
    where k = B/T. Implemented by setting the phase per Eq. (21).
    Returns H on the np.fft.fftfreq grid (0.., negatives at end).
    """
    k = B / T
    f = np.fft.fftfreq(Nfft, d=1.0/fs)          # Hz, symmetric (0.., -..)
    fb = np.abs(f)                               # use |f| for beat freq
    if fb_max is None:
        fb_max = fs/2                            # Nyquist (or choose your plot band)
    # Phase from the integral of -tau_GD (paper Eq. (21) re-arranged):
    # ϕ(fb) = -2π/k * ( fb_max*fb - 0.5*fb**2 )
    phi = -(2*np.pi/k)*(fb_max*fb - 0.5*(fb**2))
    H = np.exp(1j*phi)                           # |H|=1 all-pass
    return H

def apply_group_delay_allpass(x_time, H):
    X = np.fft.fft(x_time, n=H.size)
    y = np.fft.ifft(X * H)
    return y

def build_binary_code(Lc, family="random", seed=None):
    rng = np.random.default_rng(seed)
    if family == "random":
        return rng.choice([1.0, -1.0], size=Lc).astype(np.float32)
    # You can plug in Kasami/ZCZ here if you like (paper uses both). :contentReference[oaicite:3]{index=3}
    return rng.choice([1.0, -1.0], size=Lc).astype(np.float32)

def make_decoder_vector(Nfast, code_bits):
    """
    Make C*(t - τmax): right-align the Lc chips so the LAST chip ends at the last fast-time sample.
    """
    Lc = len(code_bits)
    spc = max(1, Nfast // Lc)                    # integer samples per chip
    dec = np.repeat(code_bits, spc)
    # Right-align to the end (τmax) and pad/crop to Nfast
    if dec.size < Nfast:
        pad = Nfast - dec.size
        dec = np.concatenate([np.ones(pad, dtype=dec.dtype), dec])
    else:
        dec = dec[-Nfast:]
    return dec                                   # real ±1; conjugate is same

# Small utility to extract the last-chirp fast-time you already build in update()
def slice_last_chirp_time(sum_data, start_offset_samples, good_ramp_samples, num_samples_frame):
    start_index = start_offset_samples
    stop_index  = start_index + good_ramp_samples
    return sum_data[start_index:stop_index].copy()
