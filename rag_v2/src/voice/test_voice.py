from .microphone import MicrophoneRecorder
from .speech_to_text import SpeechToText


def main():

    recorder = MicrophoneRecorder()

    stt = SpeechToText(
        model_name="base"
    )

    audio_file = recorder.record(
        duration=5,
        output_file="input.wav"
    )

    text = stt.transcribe_file(
        audio_file
    )

    print("\n" + "=" * 70)
    print("TRANSCRIBED TEXT")
    print("=" * 70)

    print(text)


if __name__ == "__main__":
    main()