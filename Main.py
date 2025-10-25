import AudioDSPLib
from scipy.io import wavfile
import soundfile as sf
import numpy as np

def main():
    # Define System Parameters
    SAMPLE_RATE = 44100
    SWEEP_LENGTH = 5

    # Generate Sine Sweep
    t, sineSweep = AudioDSPLib.sineSweep(SWEEP_LENGTH, SAMPLE_RATE)

    # Initialize Variables
    impulseResponse = None

    # Begin User Interface
    print("------------------------------")
    print("Welcome to Reverb Capture Pro!")
    print("------------------------------")
    print()
    print("Enter a command to begin (type help for list of commands)")

    while True:
        operation = input()

        if operation == "quit":
            quit()

        elif operation == "help":
            print("COMMANDS:" + "\n"
                  "capture   -> Play sine sweep and capture impulse response" + "\n"
                  "save      -> Save impulse response to file" + "\n"
                  "graph     -> Creates a graph of the impulse response" + "\n"
                  "convolve  -> Test captured impulse response with a wav file")

        elif operation == "capture":
            recordedAudio = AudioDSPLib.playAndRecord(sineSweep, SAMPLE_RATE)
            impulseResponse = AudioDSPLib.deconvolve(sineSweep, recordedAudio)
            print("Successfully Captured Impulse Response!")

        elif operation == "save":
            if impulseResponse is None:
                print("Nothing to Save! Please Capture First...")
            else:
                filename = input("Please enter file name: ")
                sf.write(filename + ".wav", impulseResponse, SAMPLE_RATE, subtype='PCM_24')
                print("Saved to file " + filename + ".wav")

        elif operation == "graph":
            if impulseResponse is None:
                print("Nothing to Graph! Please Capture First...")
            else:
                t_lim = float(input("Enter stop time value: "))
                print("Graphing Impulse Response...")
                AudioDSPLib.graphAudio(impulseResponse, SAMPLE_RATE, t_lim)

        elif operation == "convolve":
            if impulseResponse is None:
                print("No Impulse Response! Please Capture First...")
            else:
                filename = input("Please enter audio file name: ")
                # Add try except here
                fileSampleRate, audioFile = wavfile.read(filename)
                if fileSampleRate != SAMPLE_RATE:
                    print("Different Sample Rates!")
                    print("System:", SAMPLE_RATE)
                    print("File:", fileSampleRate)
                else:
                    convolvedAudio = AudioDSPLib.convolve(audioFile, impulseResponse)
                    sf.write("Convolved_Audio.wav", convolvedAudio, SAMPLE_RATE, subtype='PCM_24')
                    print('Saved to file "Convolved_Audio.wav"')

        else:
            print("Invalid Command! (type help for list of commands)")


def runDeconvolve():
    fs, outputAudio = wavfile.read("SineOutput.wav")
    fs2, inputAudio = wavfile.read("SineInput.wav")
    # inputAudio, fs2 = librosa.load("Audio Track.mp3", sr=None, mono=False)
    # inputAudio = inputAudio.T

    IR = AudioDSPLib.deconvolve(inputAudio, outputAudio)

    IR_int16 = (IR * 32767).astype(np.int16)
    wavfile.write("ImpulseResponse.wav", fs, IR_int16)


def testConvolve():
    fs, audio1 = wavfile.read("Sample4.wav")
    fs2, audio2 = wavfile.read("IR3.wav")

    outputAudio = AudioDSPLib.convolve(audio1, audio2)
    outputAudio_int16 = (outputAudio * 32767).astype(np.int16)
    wavfile.write("RoomAudio.wav", fs, outputAudio_int16)


def filterTest():
    fs, audio = wavfile.read("CRS.wav")
    # filtered = AudioDSPLib.fftLowPass(audio, 1000, fs)
    filtered = AudioDSPLib.fftHighPass(audio, 1000, fs)
    sf.write("FilteredAudio.wav", filtered, fs, subtype='PCM_24')


if __name__ == "__main__":
    # main()
    # runDeconvolve()
    testConvolve()
    # filterTest()