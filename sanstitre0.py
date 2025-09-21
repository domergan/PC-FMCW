# Complex-signal case: no Hermitian symmetry is required.
# We apply the same phase law to *signed* frequencies.
import numpy as np
import matplotlib.pyplot as plt

def design_group_delay_allpass_general(N, fs, K, f_max, for_real_signal=True):
    """
    Frequency-domain all-pass with |H|=1 and phase:
        phase(H(f)) = (2*pi*f)/K * (f/2 - f_max)
    If for_real_signal=True, enforce Hermitian symmetry so IFFT is real.
    If False (complex signals), apply the phase to signed frequencies directly.
    """
    f = np.fft.fftfreq(N, d=1.0/fs)  # signed frequencies (Hz)
    if for_real_signal:
        f_abs = np.abs(f)
        phi_pos = (2*np.pi*f_abs)/K * (0.5*f_abs - f_max)
        phi = np.where(f >= 0, phi_pos, -phi_pos)
        # force self-conjugate bins to phase 0 for real outputs
        phi[np.isclose(f, 0.0)] = 0.0
        if N % 2 == 0:
            phi[np.isclose(np.abs(f), fs/2)] = 0.0
    else:
        # Complex signal: no symmetry enforcement. Use signed f directly.
        phi = (2*np.pi*f)/K * (0.5*f - f_max)
        # DC is automatically zero; Nyquist can be anything
    H = np.exp(1j * phi)
    return H, f, phi


fs = 2e6
T = 0.5e-3
t = np.arange(0.0, T, 1.0/fs)
N = len(t)

# A complex burst (analytic-like): Gaussian envelope times complex exponentials
env = np.exp(-0.5*((t - 0.20)/0.04)**2)
x = env * (np.exp(1j*2*np.pi*(100e3)*t) + 0.8*np.exp(1j*2*np.pi*(150e3)*t))

# Filter parameters
f_max = 300e3   # Hz
K = (500e6)/(500e-6)     # Hz/s

H, f, phi = design_group_delay_allpass_general(N, fs, K, f_max, for_real_signal=False)

# Apply in the FFT domain
X = np.fft.fft(x)
Y = H * X
y = np.fft.ifft(Y)  # keep complex

# Sanity: all-pass should preserve magnitudes bin-wise
mag_ratio = np.linalg.norm(np.abs(Y)) / np.linalg.norm(np.abs(X))
print(f"Magnitude preservation (||Y||/||X||): {mag_ratio:.6f} (should be ~1.0)")

# Group delay from derivative (works for signed f too)
phi_unwrap = np.unwrap(np.angle(np.exp(1j*phi)))
dphi_df = np.gradient(phi_unwrap, f)
tau_numeric = -dphi_df / (2*np.pi)
tau_formula = (f_max - f) / K

# --- Plots ---
plt.figure()
f_shift = np.fft.fftshift(f)
H_shift = np.fft.fftshift(H)
plt.plot(f_shift, np.abs(H_shift))
plt.title('|H(f)| for complex-signal design (should be 1)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('|H(f)|')
plt.grid(True)

plt.figure()
plt.plot(f, tau_formula, label='Target τ(f)')
plt.plot(f, tau_numeric, linestyle='--', label='Numeric τ from phase')
plt.title('Group delay vs signed frequency (complex case)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('τ(f) [s]')
plt.legend()
plt.grid(True)

plt.figure()
X_shift = np.fft.fftshift(np.abs(X))
Y_shift = np.fft.fftshift(np.abs(Y))
plt.plot(f_shift, X_shift, label='|X(f)|')
plt.plot(f_shift, Y_shift, linestyle='--', label='|Y(f)|')
plt.title('Spectral magnitudes (all-pass, complex signal)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.legend()
plt.grid(True)

plt.figure()
plt.plot(t, np.real(x), label='Re{x}')
plt.plot(t, np.imag(x), label='Im{x}')
plt.plot(t, np.real(y), label='Re{y} filtered')
plt.plot(t, np.imag(y), label='Im{y} filtered')
plt.title('Time-domain (complex signal): real/imag parts')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)

plt.show()
