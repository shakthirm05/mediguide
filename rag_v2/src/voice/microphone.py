import sounddevice as sd
from scipy.io.wavfile import write


class MicrophoneRecorder:

    def __init__(
        self,
        sample_rate=16000
    ):

        self.sample_rate = sample_rate

    def record(
        self,
        duration=5,
        output_file="input.wav"
    ):

        print(
            f"\nRecording for {duration} seconds..."
        )

        audio = sd.rec(
            int(
                duration *
                self.sample_rate
            ),
            samplerate=self.sample_rate,
            channels=1,
            dtype="int16"
        )

        sd.wait()

        write(
            output_file,
            self.sample_rate,
            audio
        )

        print(
            f"Audio saved to {output_file}"
        )

        return output_file