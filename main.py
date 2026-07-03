import numpy as np
import scipy.linalg
from matplotlib import pyplot as plt
from run_RK4 import run_rk4_chirp, NonlinearElement, NL_CUBIC, NL_HERTZ, NL_PIECEWISE, NL_CUBIC

from pre_process import *
# ============================================================================
# System definition
# ============================================================================


# Uncomment for 1-DOF Duffing oscillator
M = np.array([1])
C = np.array([0.1])
K = np.array([1])

#Parameters for PWL : (gap, slopes)
nl_elements = [NonlinearElement(dof_i=0, dof_j=-1, nl_type=NL_CUBIC, params=(1.0,))]
fext  = np.array([1.0]) # Uncomment for 1-DOF Duffing oscillator
f_amp = 2.0 # Forcing amplitude (N)

# Sweep parameters
chirp_type = 0  # 0 = linear chirp, 1 = log chirp, 2 = constant frequency
f0 = 0.01/(2*np.pi)     # Start frequency (Hz)
f1 = 10/(2*np.pi)     # End frequency (Hz), ignored if chirp_type == 2
T  = 100000 # Sweep duration (s)
sweep_rate = (f1 - f0) / T

# Sampling
f_s = 200.0 / (2.0 * np.pi)
ts  = 1/f_s

# Initial conditions (manual)
q0 = np.array([0.0])   # Initial displacement
qd0 = np.array([0.0])  # Initial velocity


def launch_simulation():
    """Launch RK4 integration and return (t_arr, y_arr)."""
    return run_rk4_chirp(M, C, K, fext, f_amp, f0, f1, T, ts, chirp_type, nl_elements=nl_elements, q0=q0, qd0=qd0)



if __name__ == "__main__":
    t_arr, y_arr = launch_simulation()

    if chirp_type == 0:
        phase = 2.0 * np.pi * (f0 * t_arr + 0.5 * sweep_rate * t_arr**2)
        freq  = f0 + sweep_rate * t_arr                    # instantaneous frequency (Hz)
    elif chirp_type == 1:
        phase = 2.0 * np.pi * f0 * T / np.log(f1 / f0) * (np.exp(np.log(f1 / f0) * t_arr / T) - 1.0)
        freq  = f0 * (f1 / f0) ** (t_arr / T)             # instantaneous frequency (Hz)
    elif chirp_type == 2:
        phase = 2.0 * np.pi * f0 * t_arr
        freq  = np.full_like(t_arr, f0)                   # constant frequency (Hz)
    else:
        raise ValueError("chirp_type must be 0 (linear), 1 (log), or 2 (constant frequency)")
    omega_exc = 2.0 * np.pi * freq   


    ndofs = M.shape[0]

    q  = y_arr[:, :ndofs]   # displacement  (N, ndof)
    qd = y_arr[:, ndofs:]   # velocity       (N, ndof)

    #e.g. plot the displacement of the first DOF
    k = np.arange(1, 1000)
    t_0 = (-f0 + np.sqrt(f0**2 + sweep_rate * k)) / sweep_rate

    plt.figure()
    if chirp_type == 2:
        plt.plot(t_arr, q[:, 0])
        plt.xlabel("Time (s)")
    else:
        plt.plot(omega_exc, q[:, 0])
        plt.xlabel("Excitation Frequency (rad/s)")
    plt.ylabel("Displacement (m)")
    plt.title("Displacement of DOF 1")
    plt.grid()

    # # #Basic FFT plot of the response
    # # plt.figure()
    # # from scipy.fft import fft, fftfreq
    # # N = len(t_arr)

    # # yf = fft(q[:, 0])
    # # xf = fftfreq(N, ts)[:N//2]

    # # print(np.shape(xf))

    # # plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
    # # plt.grid()

    # #Basic spectrogram of the response
    # fig, ax = plt.subplots()
    # NFFT = 128*64
    # spec, freqs, t_spec, im = ax.specgram(q[:, 0], NFFT=NFFT, Fs=1/ts, noverlap=NFFT//10 * 9, cmap='viridis')
    # eps = np.finfo(float).eps
    # spec_db = 10.0 * np.log10(spec + eps)

    # # Map each spectrogram time bin to excitation frequency (Hz)
    # if chirp_type == 0:
    #     exc_freq_spec = f0 + sweep_rate * t_spec
    # elif chirp_type == 1:
    #     exc_freq_spec = f0 * (f1 / f0) ** (t_spec / T)
    # else:
    #     exc_freq_spec = np.full_like(t_spec, f0)

    # ax.cla()
    # ax.pcolormesh(exc_freq_spec, freqs, spec_db, shading="auto", cmap="viridis")

    # slopes = [0.5, 1.5, 2.5]
    # x_line = np.array([exc_freq_spec[0], exc_freq_spec[-1]])
    # for k in slopes:
    #     ax.plot(x_line, k * x_line, color='white', linestyle='--', linewidth=1, label=f'k={k}')

    # ax.set_xlabel("Excitation Frequency (Hz)")
    # ax.set_ylabel("Frequency (Hz)")
    # ax.set_title("Spectrogram")
    # ax.set_xlim(exc_freq_spec[0], exc_freq_spec[-1])
    # ax.set_ylim(0, max(freqs))
    # ax.legend(fontsize=7, loc='upper left')
    # fig.colorbar(ax.collections[0], ax=ax, label="Power/Frequency (dB/Hz)")
    # print("Spectrogram shape:", spec.shape)


    # # PSD extracted as time-average of the spectrogram power
    # psd = spec.mean(axis=1)  # shape: (n_freqs,)
    # plt.figure()
    # plt.semilogy(freqs, psd)
    # plt.xlabel("Frequency (Hz)")
    # plt.ylabel("PSD")
    # plt.title("PSD from spectrogram")
    # plt.grid()

    # # PSD along each harmonic line y = k * x
    # # For each slope k, average spec bins within +/- margin_bins of the nearest bin
    # margin_bins = 5  # number of frequency bins on each side to include
    # fig_lines, ax_lines = plt.subplots()
    # for k in slopes:
    #     target_freqs = k * exc_freq_spec          # shape: (n_times,)
    #     center_indices = np.searchsorted(freqs, target_freqs)
    #     center_indices = np.clip(center_indices, 0, len(freqs) - 1)
    #     # pick the nearest bin (searchsorted returns insertion point, check left neighbour too)
    #     left = np.clip(center_indices - 1, 0, len(freqs) - 1)
    #     use_left = np.abs(freqs[left] - target_freqs) < np.abs(freqs[center_indices] - target_freqs)
    #     center_indices = np.where(use_left, left, center_indices)

    #     in_range = (target_freqs >= freqs[0]) & (target_freqs <= freqs[-1])
    #     t_indices = np.arange(len(exc_freq_spec))

    #     # Collect and average bins within the margin window
    #     window_slices = [
    #         np.clip(center_indices + offset, 0, len(freqs) - 1)
    #         for offset in range(-margin_bins, margin_bins + 1)
    #     ]
    #     stacked = np.stack([spec[idx, t_indices] for idx in window_slices], axis=0)
    #     line_power = np.mean(stacked, axis=0)
    #     line_power = np.where(in_range, line_power, np.nan)

    #     ax_lines.semilogy(exc_freq_spec, line_power, label=f'k={k}')

    # ax_lines.set_xlabel("Excitation Frequency (Hz)")
    # ax_lines.set_ylabel("PSD along line")
    # ax_lines.set_title("PSD of pixels intersected by harmonic lines")
    # ax_lines.legend(fontsize=7)
    # ax_lines.grid()
    # plt.ylim(bottom=1e-8)  # Adjust as needed for visibility
    plt.show()
