##############################################################################
# File: ConvolutionFilter.py
# Authors: Landon Bates, Ben Polzin
# Reads in an impulse response file and an audio file and performs
# a convolution filter
###############################################################################
import numpy as np
import scipy

# Define Files
IRfile = "Test Files/Impulse Responses/Rockville_IR5.wav"
audiofile = "Test Files/Dry Audio/American_Idiot.wav"
outfile = "RockvilleConvolved.wav"

# Read in Impulse Response
samplerateIR, IR = scipy.io.wavfile.read(IRfile)
IR = IR / np.amax(np.abs(IR))

# If IR is mono, convert to stereo
if IR.ndim == 1:
    IR = np.vstack([IR, IR])
    IR = np.transpose(IR)

# Read in audio input data
samplerate, audio_in = scipy.io.wavfile.read(audiofile)
audio_in = audio_in / np.amax(np.abs(audio_in))

# If audio is mono, convert to stereo
if audio_in.ndim == 1:
    audio_in = np.vstack([audio_in, audio_in])
    audio_in = np.transpose(audio_in)

# Pad the IR with zeros so it is the same length as the input
IRpad = np.zeros((len(audio_in), 2))  # Assume audio is longer
IRpad[:len(IR)] = IR

# Convolve the audio input with the IR
IR_conv = scipy.signal.fftconvolve(audio_in, IRpad, mode='full', axes=0)

# Normalize the convolved audio with the max value
IR_conv = IR_conv / (np.amax(np.abs(IR_conv)))

# Save convolved audio into a .wav file
scipy.io.wavfile.write(outfile, samplerate, IR_conv)
