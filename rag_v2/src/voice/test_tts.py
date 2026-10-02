from .text_to_speech import TextToSpeech


def main():

    tts = TextToSpeech()

    text = (
        "Hello. Welcome to MediGuide General Hospital. "
        "How can I help you today?"
    )

    print("\nGenerating speech...")

    output = tts.speak(text)

    print(f"\nAudio generated: {output}")


if __name__ == "__main__":
    main()