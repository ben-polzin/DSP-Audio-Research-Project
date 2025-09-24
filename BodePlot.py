from scipy.signal import freqz, savgol_filter
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator
from ConvolutionLibrary import normalize

# Read in Impulse Response
#sample_rate, IR = wavfile.read("Test Files/Impulse Responses/Marshall_IR501.wav")
sample_rate, IR = wavfile.read("Test Files/Impulse Responses/Rockville_IR5.wav")

# Convert to Mono
if IR.ndim > 1:
    IR = IR[:, 0]

# Normalize Audio
IR = normalize(IR)

# Compute Frequency Response
w, h = freqz(IR, worN=8000, fs=sample_rate)
h = normalize(h)

# Define custom formatter for frequency labels
def frequency_formatter(x, pos):
    if x >= 1000:
        return f"{x / 1000:.1f} kHz"
    else:
        return f"{x:.0f} Hz"

# Create Plot
plt.figure(figsize=(12, 6))

# Magnitude plot
#plt.subplot(2, 1, 1)
plt.semilogx(w, 20 * np.log10(np.abs(h)))
plt.title("Bode Plot")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.grid()
plt.xlim(20, 15000)
plt.ylim(-50, 5)
plt.gca().xaxis.set_major_formatter(FuncFormatter(frequency_formatter))  # Apply custom formatter

# Configure y-axis ticks (magnitude axis)
ax = plt.gca()
ax.yaxis.set_major_locator(MultipleLocator(5))  # Major ticks every 5 dB
ax.yaxis.set_minor_locator(MultipleLocator(1))  # Minor ticks every 1 dB

plt.tight_layout()
plt.show()


# Phase plot
#plt.subplot(2, 1, 2)
#plt.semilogx(w, np.angle(h, deg=True))
#plt.xlabel("Frequency (Hz)")
#plt.ylabel("Phase (degrees)")
#plt.grid()
#plt.xlim(20, 15000)
#plt.gca().xaxis.set_major_formatter(FuncFormatter(frequency_formatter))  # Apply custom formatter
