import numpy as np
from scipy.fft import fft, ifft


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


def deconvolve(wetAudio, dryAudio):
    wetAudio = normalize(wetAudio)
    dryAudio = normalize(dryAudio)

    # Convert Mono to Stereo
    if wetAudio.ndim == 1:
        wetAudio = toStereo(wetAudio)

    if dryAudio.ndim == 1:
        dryAudio = toStereo(dryAudio)

    # Match Lengths of Arrays by Padding with Zeros
    wetAudio, dryAudio = lengthMatch(wetAudio, dryAudio)

    # DeConvolve Audio
    wetAudio_left, wetAudio_right = np.split(wetAudio, 2, axis=1)
    dryAudio_left, dryAudio_right = np.split(dryAudio, 2, axis=1)

    wetAudio_left = np.transpose(wetAudio_left)[0]
    wetAudio_right = np.transpose(wetAudio_right)[0]
    dryAudio_left = np.transpose(dryAudio_left)[0]
    dryAudio_right = np.transpose(dryAudio_right)[0]

    deconvolvedAudio_left = ifft(fft(wetAudio_left) / fft(dryAudio_left)).real
    deconvolvedAudio_right = ifft(fft(wetAudio_right) / fft(dryAudio_right)).real

    deconvolvedAudio_left = normalize(deconvolvedAudio_left)
    deconvolvedAudio_right = normalize(deconvolvedAudio_right)

    deconvolvedAudio = np.vstack([deconvolvedAudio_left, deconvolvedAudio_right])
    return np.transpose(deconvolvedAudio)
