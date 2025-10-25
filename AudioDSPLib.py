import numpy as np
import pyaudio
from scipy.fft import fft, ifft
from scipy.signal import fftconvolve, chirp
import matplotlib.pyplot as plt
import pandas as pd


def normalize(audio):
    return audio / np.max(np.abs(audio))


def toStereo(audio):
    audio = np.vstack([audio, audio])
    return np.transpose(audio)


def lengthMatch(array1, array2):
    if len(array1) > len(array2):
        array2_pad = np.zeros((len(array1), 2))
        array2_pad[:len(array2)] = array2
        array2 = array2_pad

    if len(array2) > len(array1):
        array1_pad = np.zeros((len(array2), 2))
        array1_pad[:len(array1)] = array1
        array1 = array1_pad

    return array1, array2


def sineSweep(length, sampleRate=48000, start=20, stop=20000):
    """Outputs an exponential sine sweep of the given length"""
    t = np.linspace(0, length, int(length * sampleRate))
    sweep = chirp(t, start, length, stop, method="logarithmic")
    return t, sweep


def getOutputDevices(sampleRate=48000):
    """Return a list of audio output devices"""
    p = pyaudio.PyAudio()
    devices = []

    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info["maxOutputChannels"] > 0 and info["defaultSampleRate"] == sampleRate:
            devices.append((i, info["name"], info["hostApi"]))
    p.terminate()
    return devices


def getInputDevices(sampleRate=48000):
    """Return a list of audio input devices"""
    p = pyaudio.PyAudio()
    devices = []

    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info["maxInputChannels"] > 0 and info["defaultSampleRate"] == sampleRate:
            devices.append((i, info["name"], info["hostApi"]))
    p.terminate()
    return devices


def playAudio(audio, sampleRate=48000, device=None):
    """Plays audio through speakers"""
    # Normalize Audio
    audio = normalize(audio)

    # Check Channels
    if audio.ndim == 1:
        channels = 1
    elif audio.ndim == 2 and audio.shape[1] == 2:
        channels = 2
    else:
        raise ValueError("Audio must be 1D (mono) or 2D with shape (N,2) (stereo).")

    # Convert audio to 16-bit PCM
    audioPCM = (audio * 32767).astype(np.int16).tobytes()

    # Open Stream
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=channels,
                    rate=sampleRate,
                    output=True,
                    output_device_index=device)

    # Play Audio
    stream.write(audioPCM)

    # Close Stream
    stream.stop_stream()
    stream.close()
    p.terminate()


def playAndRecord(outputAudio, sampleRate=48000, outputDevice=None, inputDevice=None):
    """
    ### Want to make this record for longer to account for reverb trail ###

    Plays audio through speakers while simultaneously recording from microphone.
    Returns the recorded audio as a NumPy array.
    """
    # Normalize Audio
    outputAudio = normalize(outputAudio)

    # Check Channels
    if outputAudio.ndim == 1:
        outputChannels = 1
    elif outputAudio.ndim == 2 and outputAudio.shape[1] == 2:
        outputChannels = 2
    else:
        raise ValueError("Audio must be 1D (mono) or 2D with shape (N,2) (stereo).")

    # Convert audio to 16-bit PCM
    outputAudioPCM = (outputAudio * 32767).astype(np.int16).tobytes()

    # Recording Setup
    chunk = 1024  # Frames per buffer

    # Open Stream
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=outputChannels,  # INPUT AND OUTPUT CHANNELS MUST BE THE SAME
                    rate=sampleRate,
                    output=True,
                    output_device_index=outputDevice,
                    input=True,
                    input_device_index=inputDevice,
                    frames_per_buffer=chunk)

    # Create Buffers
    recordedFrames = []
    bytesPerFrame = 2 * outputChannels  # int16 = 2 bytes per channel

    # Play & Record Audio
    for i in range(0, len(outputAudioPCM), chunk * bytesPerFrame):
        playChunk = outputAudioPCM[i:i + chunk * bytesPerFrame]
        stream.write(playChunk)
        recordChunk = stream.read(chunk)
        recordedFrames.append(recordChunk)

    # Close Stream
    stream.stop_stream()
    stream.close()
    p.terminate()

    # Convert Recorded Data to Float Array
    inputAudio = np.frombuffer(b''.join(recordedFrames), dtype=np.int16)
    inputAudio = inputAudio.astype(float) / 32767.0

    return inputAudio


def convolve(x_t, h_t):
    """
    Performs a Convolution of an input function x_t and an impulse response h_t.
    Designed for use with wav format audio files.
    """

    # Make Arrays 2D
    if x_t.ndim == 1:
        x_t = toStereo(x_t)
    if h_t.ndim == 1:
        h_t = toStereo(h_t)

    # Normalize Inputs
    x_t = normalize(x_t)
    h_t = normalize(h_t)

    # Store Length of Arrays
    x_tLength = len(x_t)
    h_tLength = len(h_t)

    # Match Lengths
    x_t, h_t = lengthMatch(x_t, h_t)

    # Convolve
    y_t = fftconvolve(x_t, h_t, mode='full', axes=0)

    # Normalize Output
    y_t = normalize(y_t)

    # Trim Output
    y_t = y_t[:(x_tLength + h_tLength)]

    return y_t


def deconvolve(x_t, y_t):
    """
    Performs a Deconvolution of output y_t and input x_t, returning the impulse response h_t.
    Designed for use with wav format audio files.
    """

    # Make Arrays 2D
    if y_t.ndim == 1:
        y_t = toStereo(y_t)
    if x_t.ndim == 1:
        x_t = toStereo(x_t)

    # Normalize Inputs
    y_t = normalize(y_t)
    x_t = normalize(x_t)

    # Match Lengths
    y_t, x_t = lengthMatch(y_t, x_t)

    # Split into Left and Right Channels
    y_tLeft, y_tRight = np.split(y_t, 2, axis=1)
    x_tLeft, x_tRight = np.split(x_t, 2, axis=1)
    y_tLeft = np.transpose(y_tLeft)[0]
    y_tRight = np.transpose(y_tRight)[0]
    x_tLeft = np.transpose(x_tLeft)[0]
    x_tRight = np.transpose(x_tRight)[0]

    # Zero Pad Signals to Avoid Circular Convolution Effects
    N = len(x_tLeft) + len(y_tLeft) - 1

    X_fLeft = fft(x_tLeft, n=N)
    Y_fLeft = fft(y_tLeft, n=N)
    X_fRight = fft(x_tRight, n=N)
    Y_fRight = fft(y_tRight, n=N)

    # Deconvolve
    eps = 1e-10
    h_tLeft = ifft(Y_fLeft / (X_fLeft + eps)).real
    h_tRight = ifft(Y_fRight / (X_fRight + eps)).real

    # Normalize Output
    h_tLeft = normalize(h_tLeft)
    h_tRight = normalize(h_tRight)

    # Recombine Left and Right Channels
    h_t = np.transpose(np.vstack([h_tLeft, h_tRight]))

    # Trim Output
    h_t = h_t[:960000]  # Temporary Code

    return h_t


def farinaDeconvolve(x_t, y_t):
    """
    Performs a linear deconvolution using the Farina method
    Inputs:
    x_t - Input Sine Sweep
    y_t - Recorded Sine Sweep

    Outputs:
    h_t - System Impulse Response
    """

    # Find Inverse Filter
    inv = 0

    # Convolve with Inverse Filter
    h_t = fftconvolve(y_t, inv, mode='full')

    # Normalize Impulse Response
    h_t = normalize(h_t)

    return h_t


def graphAudio(audio, sampleRate, t_lim=None):
    """Inputs Audio and SampleRate and creates a graph using Matplotlib"""
    t0 = 0
    tf = len(audio) / sampleRate
    t = np.linspace(t0, tf, len(audio))

    plt.figure(figsize=(13, 7))
    plt.plot(t, audio)
    plt.title("Audio Graph")
    plt.xlabel("Time (s)")
    plt.ylabel("Magnitude")
    plt.grid()

    if t_lim is not None:
        plt.xlim(0, t_lim)
    else:
        plt.xlim(0, tf)

    plt.show()



def fftLowPass(audio, cutoff_hz, sampleRate):
    # --- Check that the input is stereo ---
    if audio.ndim != 2 or audio.shape[1] != 2:
        audio = toStereo(audio)

    # --- Normalize input ---
    audio = normalize(audio)

    # --- Frequency axis ---
    N = audio.shape[0]
    freqs = np.fft.fftfreq(N, d=1 / sampleRate)
    mask = np.abs(freqs) <= cutoff_hz  # 1D mask for frequencies below cutoff

    # --- Process left and right channels separately ---
    X_left = fft(audio[:, 0])
    X_right = fft(audio[:, 1])

    X_left_filtered = X_left * mask
    X_right_filtered = X_right * mask

    # --- Inverse FFT to return to time domain ---
    y_left = ifft(X_left_filtered).real
    y_right = ifft(X_right_filtered).real

    # --- Recombine and normalize ---
    y_t = np.column_stack((y_left, y_right))
    y_t = normalize(y_t)

    return y_t


def fftHighPass(audio, cutoff_hz, sampleRate):

    # --- Check that the input is stereo ---
    if audio.ndim != 2 or audio.shape[1] != 2:
        audio = toStereo(audio)

    # --- Normalize input ---
    audio = normalize(audio)

    # --- Frequency axis ---
    N = audio.shape[0]
    freqs = np.fft.fftfreq(N, d=1 / sampleRate)
    mask = np.abs(freqs) >= cutoff_hz  # Keep frequencies above cutoff

    # --- Process left and right channels separately ---
    X_left = fft(audio[:, 0])
    X_right = fft(audio[:, 1])

    X_left_filtered = X_left * mask
    X_right_filtered = X_right * mask

    # --- Inverse FFT to return to time domain ---
    y_left = ifft(X_left_filtered).real
    y_right = ifft(X_right_filtered).real

    # --- Recombine and normalize ---
    y_t = np.column_stack((y_left, y_right))
    y_t = normalize(y_t)

    return y_t



# Additional Functions For Analysis

def readCSV(filename, x_name="second", y_name="Volt"):
    dataFrame = pd.read_csv(filename)
    x = np.array(dataFrame[x_name], dtype=float)
    y = np.array(dataFrame[y_name], dtype=float)
    return x, y


def plotAvsM(t_M, v_M, t_A, v_A, title, xLabel="Time (s)", yLabel="Voltage (V)"):
    """Plot measured solution vs analytical solution"""
    plt.figure(figsize=(8, 5))
    plt.plot(t_A, v_A, color="blue", label="Analytical")
    plt.plot(t_M, v_M, color="red", label="Measured")
    plt.title(title)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.xlim(0, 0.001)
    plt.legend()
    plt.grid()
    plt.show()