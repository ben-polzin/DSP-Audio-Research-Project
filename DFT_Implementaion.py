
import numpy as np
from scipy.io import wavfile
from scipy.fft import fft

#--- Under Development --#
def DFT(data, samplerate):
    N = len(data)
    f_resolution = samplerate / N
    ftL = []
    ftR = []
    for ii, (L, R) in enumerate(data):
        f = f_resolution * ii
        sum_sin1 = 0
        sum_cos1 = 0
        for i in range(N):
        
            sum_sin1 += L*np.sin(2*np.pi*f*i/N)
            sum_cos1 += R*np.cos(2*np.pi*f*i/N)

        ftL.append(np.sqrt( ((sum_cos1/N)**2) + ((sum_sin1/N)**2) ))
    #return [list(pair) for pair in zip(ftL, ftR)]
    return ftL
            

#wav file locations
audiofile = "Test Files/Dry Audio/Song Dry.wav"

#Open and store wav files in list, and sample rate {samples per second}
samplerate, data_stereo = wavfile.read(audiofile)
#Normalize
#data_stereo = data_stereo / np.max(np.abs(data_stereo))

data_stereo = data_stereo[:10][:10]

#print(data_stereo)

print("dft data: " + str(DFT(data_stereo, samplerate)))

fft_data = fft(data_stereo)
print("fft data: " + str(fft_data))