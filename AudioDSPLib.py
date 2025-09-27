import numpy as np
import pyaudio
from scipy.fft import fft, ifft
from scipy.signal import fftconvolve, chirp
import matplotlib.pyplot as plt


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
    """Performs a Convolution of an input function x_t and an impulse response h_t.
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

    # Match Lengths
    x_t, h_t = lengthMatch(x_t, h_t)

    # Convolve
    y_t = fftconvolve(x_t, h_t, mode='full', axes=0)

    # Normalize Output
    y_t = normalize(y_t)

    return y_t


def deconvolve(x_t, y_t):
    """Performs a Deconvolution of output y_t and input x_t, returning the impulse response h_t.
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

    # Deconvolve
    h_tLeft = ifft(fft(y_tLeft) / fft(x_tLeft)).real
    h_tRight = ifft(fft(y_tRight) / fft(x_tRight)).real

    # Normalize Output
    h_tLeft = normalize(h_tLeft)
    h_tRight = normalize(h_tRight)

    # Recombine Left and Right Channels
    h_t = np.transpose(np.vstack([h_tLeft, h_tRight]))

    return h_t


def graphAudio(audio, sampleRate):
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
    plt.show()
