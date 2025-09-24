from scipy.signal import freqz, savgol_filter
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator
from ConvolutionLibrary import normalize

# Read in Impulse Response
sample_rate, IR = wavfile.read("Test Files/Impulse Responses/Rockville_IR3.wav")

# Convert to Mono
if IR.ndim > 1:
    IR = IR[:, 0]

# Normalize Audio
IR = normalize(IR)

# Compute Frequency Response
w, h = freqz(IR, worN=8000, fs=sample_rate)
h = normalize(h)

# Smoothing parameters
smoothing_enabled = True
min_window = 21  # Minimum window size (must be odd and >= order+1)
max_window = 801  # Maximum window size (must be odd)
smoothing_order = 3  # Polynomial order (must be < window_size)


def safe_savgol_filter(data, window_size, order):
    """A wrapper around savgol_filter that handles edge cases"""
    if window_size > len(data):
        window_size = len(data) if len(data) % 2 == 1 else len(data) - 1
    if window_size <= order:
        return data  # Can't perform smoothing with this order
    return savgol_filter(data, window_size, order)


def frequency_dependent_smoothing(magnitude, frequencies, min_win, max_win, order):
    """Apply increasing smoothing with frequency"""
    n_points = len(magnitude)
    smoothed = np.zeros(n_points)

    # Calculate frequency weights on log scale
    log_freqs = np.log10(np.clip(frequencies, 20, 20000))  # Clip to valid range
    weights = (log_freqs - np.log10(20)) / (np.log10(20000) - np.log10(20))

    # Pre-calculate all window sizes
    window_sizes = np.round(min_win + (max_win - min_win) * weights).astype(int)
    window_sizes = np.clip(window_sizes, min_win, max_win)
    window_sizes = window_sizes // 2 * 2 + 1  # Ensure odd

    # Apply smoothing for each point
    for i in range(n_points):
        window_size = window_sizes[i]
        half_win = window_size // 2
        start = max(0, i - half_win)
        end = min(n_points, i + half_win + 1)

        # Get the neighborhood and apply safe smoothing
        neighborhood = magnitude[start:end]
        if len(neighborhood) >= order + 1:
            smoothed_neighborhood = safe_savgol_filter(neighborhood, window_size, order)
            # Take the center value
            smoothed[i] = smoothed_neighborhood[min(half_win, len(smoothed_neighborhood) - 1)]
        else:
            smoothed[i] = magnitude[i]  # Fallback

    return smoothed


# Apply smoothing if enabled
if smoothing_enabled:
    h_smooth = frequency_dependent_smoothing(np.abs(h), w, min_window, max_window, smoothing_order)
    h_smooth = normalize(h_smooth)
else:
    h_smooth = np.abs(h)


# Define custom formatter for frequency labels
def frequency_formatter(x, pos):
    if x >= 1000:
        return f"{x / 1000:.1f} kHz"
    else:
        return f"{x:.0f} Hz"


# Create Plot
plt.figure(figsize=(12, 6))

# Magnitude plot
plt.semilogx(w, 20 * np.log10(h_smooth))
plt.title("Bode Plot with Frequency-Dependent Smoothing")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.grid()
plt.xlim(20, 15000)
plt.ylim(-50, 5)
plt.gca().xaxis.set_major_formatter(FuncFormatter(frequency_formatter))

# Configure y-axis ticks
ax = plt.gca()
ax.yaxis.set_major_locator(MultipleLocator(5))
ax.yaxis.set_minor_locator(MultipleLocator(1))

plt.tight_layout()
plt.show()