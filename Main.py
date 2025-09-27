import AudioDSPLib


def main():
    # Define System Parameters
    sampleRate = 48000
    sweepLength = 5

    # Generate Sine Sweep
    t, sweep = AudioDSPLib.sineSweep(sweepLength, sampleRate)

    # Play Sweep and Record Response
    inputAudio = AudioDSPLib.playAndRecord(sweep, sampleRate)
    inputAudio = inputAudio[:(sampleRate * sweepLength)]  # Trim Audio, DO MORE TESTING HERE

    # Deconvolve
    impulseResponse = AudioDSPLib.deconvolve(sweep, inputAudio)

    # Save File
    AudioDSPLib.graphAudio(sweep, 48000)


    pass


if __name__ == "__main__":
    main()
