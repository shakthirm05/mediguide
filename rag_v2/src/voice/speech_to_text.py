import whisper


class SpeechToText:

    def __init__(
        self,
        model_name="base"
    ):

        print("\nLoading Whisper...")

        self.model = whisper.load_model(
            model_name
        )

        print("Whisper loaded.")

    def transcribe_file(
        self,
        audio_file
    ):

        result = self.model.transcribe(
            audio_file
        )

        return result["text"].strip()