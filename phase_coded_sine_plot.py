import numpy as np
import matplotlib.pyplot as plt

def make_tx_buffer(phases, fc, fs, num_samples, start_phase=0.0):

    phases = np.asarray(phases, dtype=float)
    M = len(phases)

    # spread phase values across the total length
    base = num_samples // M
    rem  = num_samples %  M
    counts = np.full(M, base, dtype=int)
    counts[:rem] += 1                       

    phase_per_sample = np.repeat(phases, counts)
    assert len(phase_per_sample) == num_samples

    t = np.arange(num_samples) / fs
    iq = np.exp(1j * (2*np.pi*fc*t + start_phase + phase_per_sample))
    return iq, t, phase_per_sample



fs = 2_000_000
fc = 100e3
phases = [0, np.pi, 0, np.pi]
N = 10

iq, t, phase_samples = make_tx_buffer(phases, fc, fs, N)

plt.figure()
plt.plot(t, np.real(iq))                # or np.abs(iq)
plt.title('Phase-coded sine (1000 samples total)')
plt.xlabel('Time (s)'); plt.ylabel('Amplitude'); plt.grid(True)

plt.figure()
plt.step(t, phase_samples, where='post')
plt.title('Per-sample phase offsets φ_k (radians)')
plt.xlabel('Time (s)'); plt.ylabel('Phase (rad)'); plt.grid(True)
plt.show()