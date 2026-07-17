import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from babel.dates import time_
from scipy.optimize import curve_fit
from scipy.signal import correlate, resample
from scipy.interpolate import interp1d
from scipy.interpolate import CubicSpline
import csv

#zpracovani modulovanych dat verze Unor 2026 __ zpracovani dat do bakalarky
def create_data(time_idx, value_idx, data, start, stop):
    # Select the columns
    time = data.iloc[:, time_idx]
    value = data.iloc[:, value_idx]

    # Create a mask for time within [start, stop]
    mask = (time >= start) & (time <= stop)

    # Apply the mask
    time_cut = time[mask]
    time_cut = time_cut - start
    value_cut = value[mask]

    return time_cut, value_cut


data = pd.read_csv("h148_zdroj.csv", sep=';', skiprows=5)
output_file = "h148_1.csv"

time = data.iloc[:,1]
plate = data.iloc[:,2]

start = 100
stop = 12000

f0 = 0.0045*3
f_fit=f0/3
show = 1
cutoff = 0.02

plt.plot(time, plate)
plt.xlim(start,stop)
plt.show()

plot_special=False
#------------------------------------------------------------------------------------------------------------

#snimace = [19, 20, 22, 24, 26]
snimace = [19, 20, 22, 26]
desky = [1,2,4]
tlak = [53,54]

cut = []

for i in range(len(desky)-1):
    time, value = create_data(desky[0], desky[i + 1], data, start, stop)
    cut.append(time)
    cut.append(value)

for i in range (len(snimace)-1):
    time, value = create_data(snimace[0],snimace[i+1], data, start, stop)
    cut.append(time)
    cut.append(value)

for i in range(len(tlak)-1):
    time, value = create_data(tlak[0], tlak[i + 1], data, start, stop)
    cut.append(time)
    cut.append(value)

cut = np.array(cut)
plt.plot(cut[0], cut[1])
plt.show()

#---------------------------------------------------------------------------------------------------------------

plt.figure(figsize=(10,6))


t = cut[2]
v = cut[3]
plt.plot(t, v, label=f"Signal")

plt.xlabel("Čas")
plt.ylabel("Hodnota")
plt.legend()
plt.grid(True)
plt.show()

# --- Sliding mean for irregular sampling ---
def sliding_mean_by_seconds(time, value, window_seconds):
    t = np.asarray(time)
    v = np.asarray(value)
    n = len(t)
    bg = np.empty(n)
    half = window_seconds / 2.0

    for i in range(n):
        left = np.searchsorted(t, t[i] - half, 'left')
        right = np.searchsorted(t, t[i] + half, 'right')
        bg[i] = v[left:right].mean()
    return bg


period = 1/f0
window_sec = 3 * period   # good starting point

n_signals = len(cut) // 2


# ================
# FIGURE 1: original + sliding mean
# ================
plt.figure("Original + Sliding Mean", figsize=(12, 8))

for k in range(n_signals):
    t = np.asarray(cut[2*k])
    v = np.asarray(cut[2*k+1])

    # sort (just in case)
    idx = np.argsort(t)
    t = t[idx]
    v = v[idx]

    bg = sliding_mean_by_seconds(t, v, window_sec)

    ax = plt.subplot(n_signals, 1, k+1)
    ax.plot(t, v, label=f"Signal {k} original")
    ax.plot(t, bg, '--', label="Sliding mean", linewidth=2)
    ax.legend(loc="upper right", fontsize="small")
    ax.grid(True)
    ax.set_ylabel("Value")
    if k == n_signals - 1:
        ax.set_xlabel("Time [s]")

plt.tight_layout()
plt.show()


# ================
# FIGURE 2: detrended
# ================
plt.figure("Detrended Signals", figsize=(12, 8))

labels = ["Top plate temperature","Bottom plate temperature", "Ge 08", "Ge 11", "Ge 05", "Pressure (Baratron)"]
y_axes = ["T [K]", "T [K]","T [K]","T [K]","T [K]","p [Pa]"]

for k in range(n_signals):
    t = np.asarray(cut[2*k])
    v = np.asarray(cut[2*k+1])
    idx = np.argsort(t)
    t = t[idx]
    v = v[idx]

    bg = sliding_mean_by_seconds(t, v, window_sec)
    detr = v - bg

    ax = plt.subplot(n_signals, 1, k+1)
    ax.plot(t, detr,color="blue", label=labels[k])
    ax.legend(loc="upper right", fontsize=12)
    ax.grid(True)
    ax.set_ylabel(y_axes[k], fontsize=15)
    ax.tick_params(axis="both", labelsize=13    )
    if k == n_signals - 1:
        ax.set_xlabel("Time [s]", fontsize=15)

plt.tight_layout()
#plt.savefig(
 #   r"C:\Users\sulta\Documents\Fyzika muni\Bakalarska prace\Bakalarska_prace_moje\text_prace\sliding_mean.png",
  #  dpi=300
#)
plt.show()

if plot_special == True:
    plt.figure("Detrended Signals", figsize=(8, 6))
    k = 4
    t = np.asarray(cut[2*k])
    v = np.asarray(cut[2*k+1])

    idx = np.argsort(t)
    t = t[idx]
    v = v[idx]

    bg = sliding_mean_by_seconds(t, v, window_sec)
    detr = v - bg

    plt.plot(t, detr, color="blue", label=labels[k])
    #plt.legend(loc="upper right", fontsize=13)
    plt.ylabel("Temperature [K]", fontsize=15)
    plt.xlabel("Time [s]", fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    plt.tight_layout()
    #plt.savefig(
     #   r"C:\Users\sulta\Documents\Fyzika muni\Bakalarska prace\Bakalarska_prace_moje\text_prace\sliding_mean_one.png",
    #     dpi=300
    #)
    plt.show()





def fft_lowpass_filter(t, x, cutoff_freq):

    x = np.asarray(x)
    n = len(x)

    # sampling frequency
    dt = np.mean(np.diff(t))
    fs = 1.0 / dt

    # FFT
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, d=dt)

    # create mask (1 below cutoff, 0 above)
    mask = np.ones_like(freqs)
    mask[freqs > cutoff_freq] = 0.0

    # apply mask
    X_filtered = X * mask

    # back to time domain
    x_filtered = np.fft.irfft(X_filtered, n=n)
    return x_filtered



def fft_amplitude_from_peak(signal, fs):
    N = len(signal)
    fft_vals = np.fft.rfft(signal)         # one-sided FFT
    freqs = np.fft.rfftfreq(N, 1/fs)

    # amplitude spectrum (properly scaled)
    amp = (2.0 / N) * np.abs(fft_vals)

    # find peak
    k = np.argmax(amp)
    peak_freq = freqs[k]
    peak_amp = amp[k]

    # also return complex FFT coefficient for phase reconstruction
    peak_phase = np.angle(fft_vals[k])

    return peak_freq, peak_amp, peak_phase, freqs, amp


# ================================================================
# Plot amplitude spectrum of the FFT-filtered signals + reconstruction
# ================================================================
plt.figure("Spectra (FFT Magnitude)", figsize=(12, 8))

results = []

for k in range(n_signals):

    # ------------------------
    # Loading & detrending
    # ------------------------
    t = np.asarray(cut[2*k])
    v = np.asarray(cut[2*k+1])
    idx = np.argsort(t)
    t = t[idx]
    v = v[idx]

    bg = sliding_mean_by_seconds(t, v, window_sec)
    detr = v - bg


    dt = np.mean(np.diff(t))
    fs = 1.0 / dt

    # ------------------------
    # Method II: FILTER and FFT
    # ------------------------
    detr_fft = fft_lowpass_filter(t, detr, cutoff)

    # ------------------------
    # FFT spectrum
    # ------------------------
    N = len(detr_fft)
    Y = np.fft.rfft(detr)
    freqs = np.fft.rfftfreq(N, d=dt)
    amplitude = np.abs(Y) / N * 2     # magnitude spectrum

    # Accurate FFT amplitude + frequency + phase
    peak_f, peak_A, peak_phase, f, A = fft_amplitude_from_peak(detr_fft, fs)


    # ------------------------
    # (NEW) Reconstruct pure sine wave
    # ------------------------
    reconstructed = peak_A * np.sin(2*np.pi*peak_f*t + peak_phase)

    # ------------------------
    # Plot frequency spectrum
    # ------------------------
    ax = plt.subplot(n_signals, 1, k+1)
    ax.plot(freqs, amplitude, linewidth=1.2)
    ax.set_xlim(0, cutoff * 2)
    ax.set_ylabel("Amp")
    ax.grid(True)

    if k == 0:
        ax.set_title("Amplitude spectra of FFT-filtered signals")
    if k == n_signals - 1:
        ax.set_xlabel("Frequency [Hz]")

plt.tight_layout()


def general_sin(x, a, b, c, d):
    return a*np.sin(b*x + c) + d



def sinusoidal_fit(x, y, f_fit):
    initial_guess = [max(y) - min(y), 2 * np.pi * f_fit, 0, np.mean(y)]
    popt, pcov = curve_fit(general_sin, x, y, p0=initial_guess)

    a, b, c, d = popt
    y_fit = general_sin(x, a, b, c, d)

    amplitude = abs(popt[0])
    phase_shift = popt[2]

    return y_fit, amplitude, phase_shift

# --------------------------
# Reconstruct and plot dominant sine on top of detrended data
# --------------------------
plt.figure("Reconstruction check (time domain)", figsize=(12, 8))

for k in range(n_signals):
    t = np.asarray(cut[2*k])
    v = np.asarray(cut[2*k+1])
    idx = np.argsort(t)
    t = t[idx]
    v = v[idx]

    # background and detrend (same as above)
    bg = sliding_mean_by_seconds(t, v, window_sec)
    detr = v - bg

    # ------------------------
    # Method I: Sine fitting
    # ------------------------

    y_fit, amplitude, phase_shift = sinusoidal_fit(t, detr, f_fit)

    # ------------------------
    # Method II: FFT and reconstruction
    # ------------------------

    # sampling frequency
    dt = np.mean(np.diff(t))
    fs = 1.0 / dt
    N = len(detr)

    # FILTERED SIGNAL (same as used for spectrum)
    detr_fft = fft_lowpass_filter(t, detr, cutoff)

    # FFT of the filtered signal
    Y = np.fft.rfft(detr)
    freqs = np.fft.rfftfreq(N, d=dt)

    # find peak bin (use magnitude)
    k_peak = np.argmax(np.abs(Y))
    peak_freq = freqs[k_peak]
    peak_amp = (2.0 / N) * np.abs(Y[k_peak])   # same amplitude scaling as your spectrum plot
    peak_phase = np.angle(Y[k_peak])

    print(f"Signal {k}: peak bin {k_peak}, freq = {peak_freq:.6f} Hz, amp = {peak_amp:.6e}, phase = {peak_phase:.4f} rad")
    print(f"from sin fit: amp = {amplitude:.6e}, phase = {phase_shift:.4f} rad")
    print()
    results.append(peak_freq)
    results.append(peak_amp)
    results.append(peak_phase)
    results.append(amplitude)
    results.append(phase_shift)


    # reconstruct the real sinusoid contributed by that FFT bin:
    # contribution = (2|Yk|/N) * cos(2*pi*f*t + phase)
    reconstructed = peak_amp * np.cos(2.0 * np.pi * peak_freq * t + peak_phase)

    # OPTIONAL: compute best time lag by cross-correlation (to visually align if there's a phase offset)
    # this can correct a constant time shift between detr_fft and reconstructed
    do_align = False
    if do_align:
        # cross correlate the filtered signal with the unshifted reconstructed
        cc = np.correlate(detr_fft - np.mean(detr_fft), reconstructed - np.mean(reconstructed), mode='full')
        lag_idx = np.argmax(cc) - (len(reconstructed) - 1)
        lag_time = lag_idx * dt
        # shift reconstructed by lag_time (positive lag means reconstructed should be shifted right)
        reconstructed_aligned = np.roll(reconstructed, -lag_idx)
    else:
        lag_time = 0.0
        reconstructed_aligned = reconstructed

    # Plot time-domain comparison
    ax = plt.subplot(n_signals, 1, k+1)
    ax.plot(t, detr, alpha=0.25, label=f"Signal {k} detrended (raw)")
    ax.plot(t, y_fit, label="Sin fit", linewidth=1.4)
    ax.plot(t, reconstructed, linewidth=2.0, linestyle='--', label="reconstructed (from FFT peak)")
    if do_align:
        ax.plot(t, reconstructed_aligned, linewidth=1.5, linestyle=':', label=f"recon aligned (lag {lag_time:.3f}s)")
    ax.set_ylabel("Value")
    ax.grid(True)
    ax.legend(loc="upper right", fontsize="small")
    if k == n_signals - 1:
        ax.set_xlabel("Time [s]")

plt.tight_layout()
plt.show()

# reference signal (top plate)
t_ref = np.asarray(cut[0])
v_ref = np.asarray(cut[1])

# sort reference
idx = np.argsort(t_ref)
t_ref = t_ref[idx]
v_ref = v_ref[idx]

# detrend and filter reference
bg_ref = sliding_mean_by_seconds(t_ref, v_ref, window_sec)
ref_detr = v_ref - bg_ref

dt = np.mean(np.diff(t_ref))
fs = 1.0 / dt

def get_phase_shift_uniform(t_ref, x_ref, t_sig, x_sig, period, dt):
    # Determine common time range
    t_start = max(t_ref[0], t_sig[0])
    t_end = min(t_ref[-1], t_sig[-1])
    t_uniform = np.arange(t_start, t_end, dt)

    # Interpolate both signals onto uniform time grid
    x_ref_uniform = interp1d(t_ref, x_ref, kind='linear', fill_value='extrapolate')(t_uniform)
    x_sig_uniform = interp1d(t_sig, x_sig, kind='linear', fill_value='extrapolate')(t_uniform)

    # Remove DC offset
    x_ref_z = x_ref_uniform - np.mean(x_ref_uniform)
    x_sig_z = x_sig_uniform - np.mean(x_sig_uniform)

    # Compute cross-correlation
    n = len(t_uniform)
    corr = correlate(x_sig_z, x_ref_z, mode='full')
    lags = np.arange(-n + 1, n)

    # Find lag with maximum correlation
    k = np.argmax(corr)
    lag_samples = lags[k]
    lag_seconds = lag_samples * dt

    # Convert lag to phase
    phase_rad = 2 * np.pi * (lag_seconds / period)

    return lag_seconds, phase_rad


# detrend (sliding mean only)
# Parameters
dt = 0.01  # desired uniform time step for resampling (adjust as needed)

# Detrend reference
bg_ref = sliding_mean_by_seconds(t_ref, v_ref, window_sec)
ref_detr = v_ref - bg_ref

print("Signal | Phase shift (s) | Phase shift (deg)")
print("-----------------------------------------------------------")

for k in range(1, n_signals):
    # Load and sort signal
    t = np.asarray(cut[2*k])
    v = np.asarray(cut[2*k+1])
    idx = np.argsort(t)
    t = t[idx]
    v = v[idx]

    # Detrend
    bg = sliding_mean_by_seconds(t, v, window_sec)
    detr = v - bg

    # Amplitude (optional)
    amplitude = (detr.max() - detr.min()) / 2

    # Phase shift using uniform resampling
    lag_sec, phase_rad = get_phase_shift_uniform(t_ref, ref_detr, t, detr, period, dt)
    phase_deg = np.degrees(phase_rad)

    print(f"{k:>6} {lag_sec:14.2f} | {phase_deg:14.2f}")
    results.append(lag_sec)


# Plot signals with shift applied
for k in range(1, n_signals):
    t = np.asarray(cut[2*k])
    v = np.asarray(cut[2*k+1])
    idx = np.argsort(t)
    t = t[idx]
    v = v[idx]

    # Detrend
    bg = sliding_mean_by_seconds(t, v, window_sec)
    detr = v - bg

    # Lag and shifted time
    lag_sec, _ = get_phase_shift_uniform(t_ref, ref_detr, t, detr, period, dt)
    t_shifted = t - lag_sec

    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(t_ref, ref_detr / max(ref_detr), 'k', linewidth=2, label="Top plate (reference signal)", color='black')
    plt.plot(t, detr / max(detr), alpha=0.4, label="Original signal", color='red')
    plt.plot(t_shifted, detr / max(detr), label=f"Shifted signal", color='blue')
    plt.xlabel("Time [s]", fontsize=15)
    plt.ylabel("Normalized signal", fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(fontsize=15)
    #if plot_special==True:
       # if k==4:
            #plt.savefig(
             #   r"C:\Users\sulta\Documents\Fyzika muni\Bakalarska prace\Bakalarska_prace_moje\text_prace\cross.png",
              #    dpi=300
               # )
    plt.show()

with open(output_file, mode='w', newline='') as file:
    writer = csv.writer(file, delimiter='\t')
    writer.writerow(results)

print(f"Results saved to {output_file}")









