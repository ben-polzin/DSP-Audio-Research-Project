import numpy as np
from scipy.fft import fft, ifft
from scipy.signal import fftconvolve


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

