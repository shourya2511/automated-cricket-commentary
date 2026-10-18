"""
text_to_speech.py

Converts commentary text to speech using gTTS (Google Text-to-Speech)
and saves it as an MP3 file.

Usage:
    from text_to_speech import text_to_audio
    text_to_audio("That's a gorgeous cover drive!", "output/commentary_001.mp3")
"""

from pathlib import Path
from gtts import gTTS


def text_to_audio(text: str, output_path: str, lang: str = "en", slow: bool = False):
    """Converts text to an MP3 file at output_path. Creates parent
    directories if needed."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tts = gTTS(text=text, lang=lang, slow=slow)
    tts.save(str(output_path))
    print(f"Saved audio to {output_path}")
    return output_path


if __name__ == "__main__":
    sample_text = "That's a gorgeous cover drive, timed to perfection!"
    text_to_audio(sample_text, "data/commentary/sample_output.mp3")