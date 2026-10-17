"""Ghana Game Voice: OmniVoice-Ghana speech in labelled reference voices."""
from .tts import LANGUAGES, GameVoice, language_code
from .voices import DATASET, Voice, load_voices

__all__ = ["GameVoice", "Voice", "load_voices", "language_code", "LANGUAGES", "DATASET"]
__version__ = "0.1.0"
