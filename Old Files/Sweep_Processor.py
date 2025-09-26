###################################################
# Authors: Ben Polzin, Joseph Brower
# Inputs a Dry Sine Sweep and a Recorded Sine Sweep
# and Outputs the Impulse Response
###################################################

from AudioDSPLib import deconvolve
from scipy.io import wavfile

# Read in Files
samplerateRS, response = wavfile.read('Test Files/Sweeps/Rockville_Sweep.wav')
samplerateSweep, sweep = wavfile.read('Test Files/Sweeps/Sweep15.wav')

# Extract Impulse Response
IR = deconvolve(response, sweep)

# Output IR to a Wav File
wavfile.write("Test Files/Impulse Responses/Rockville_IR2.wav", samplerateRS, IR)
