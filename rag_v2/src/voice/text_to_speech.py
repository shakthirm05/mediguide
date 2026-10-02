from pathlib import Path
import subprocess
import sys

import sounddevice as sd
from scipy.io.wavfile import read


# ==================================================
# PATHS
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT /
    "en_US-lessac-medium.onnx"
)

DEFAULT_OUTPUT = (
    PROJECT_ROOT /
    "output_tts.wav"
)


class TextToSpeech:

    def __init__(
        self,
        model_path=MODEL_PATH
    ):

        self.model_path = Path(
            model_path
        )

        if not self.model_path.exists():

            raise FileNotFoundError(
                "\nPiper model not found:\n"
                f"{self.model_path}"
            )

        print(
            "\nPiper TTS ready."
        )


    # ==================================================
    # SYNTHESIZE
    # ==================================================

    def synthesize(
        self,
        text,
        output_file=DEFAULT_OUTPUT
    ):

        output_file = Path(
            output_file
        )


        command = [

            sys.executable,

            "-m",
            "piper",

            "-m",
            str(self.model_path),

            "-f",
            str(output_file),

            "--",

            text
        ]


        subprocess.run(
            command,
            check=True
        )


        return output_file


    # ==================================================
    # SPEAK
    # ==================================================

    def speak(
        self,
        text,
        output_file=DEFAULT_OUTPUT
    ):

        audio_file = self.synthesize(
            text=text,
            output_file=output_file
        )


        sample_rate, audio = read(
            audio_file
        )


        sd.play(
            audio,
            sample_rate
        )


        sd.wait()


        return audio_file