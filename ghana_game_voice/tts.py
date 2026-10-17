"""GameVoice: OmniVoice-Ghana speech in a labelled reference voice."""
import io
import os
import threading

import numpy as np

from .voices import DATASET, find_voice, load_voices

MODEL_REPO = "ghananlpcommunity/omnivoice-ghana"
CHECKPOINT = "checkpoint-45000"
SAMPLE_RATE = 24000

# Languages the omnivoice-ghana fine-tune was trained on: name -> OmniVoice language code.
LANGUAGES = {
    "Twi (Asante)": "twi-asante",
    "Twi (Akuapem)": "twi-akuapem",
    "Fante": "fat",
    "Ewe": "ewe",
    "Dagbani": "dag",
    "Dagaare": "dga",
    "Dangme": "ada",
    "Nzema": "nzi",
    "Hausa": "hau",
    "English": "en",
}
_ALIASES = {"twi": "twi-asante", "asante twi": "twi-asante", "akuapem twi": "twi-akuapem", "fante": "fat",
            "ewe": "ewe", "dagbani": "dag", "dagaare": "dga", "dangme": "ada", "nzema": "nzi", "hausa": "hau",
            "english": "en"}


def language_code(language):
    """Accepts a code (twi-asante), a name (Twi (Asante)) or a short alias (twi, ewe)."""
    if language in LANGUAGES.values():
        return language
    if language in LANGUAGES:
        return LANGUAGES[language]
    code = _ALIASES.get(str(language).strip().lower())
    if code:
        return code
    raise ValueError(f"unknown language {language!r}; use one of {sorted(LANGUAGES.values())}")


def _patch_torchaudio_load():
    """Read audio with soundfile (avoids torchaudio's torchcodec / CUDA-version coupling)."""
    import soundfile as sf
    import torch
    import torchaudio

    def _load(path, *args, **kwargs):
        data, sr = sf.read(path, dtype="float32", always_2d=True)
        return torch.from_numpy(data.T.copy()), sr
    torchaudio.load = _load


def trim_silence(wav, sr, floor_db=-40.0, pad=0.15):
    """Cut leading/trailing silence (frames quieter than floor_db below the peak)."""
    n = int(0.02 * sr)
    if len(wav) < n:
        return wav
    frames = wav[: len(wav) // n * n].reshape(-1, n)
    db = 20 * np.log10(np.sqrt((frames ** 2).mean(1)) + 1e-9)
    loud = np.where(db > db.max() + floor_db)[0]
    if len(loud) == 0:
        return wav
    p = int(pad * sr)
    return wav[max(0, loud[0] * n - p): min(len(wav), (loud[-1] + 1) * n + p)]


class GameVoice:
    """
    >>> gv = GameVoice()
    >>> gv.speak("Fi me dan mu seesei ara!", voice="adult male - angry", language="twi", out="line.wav")

    The model loads on first use (a GPU is strongly recommended; CPU works but is slow).
    """

    def __init__(self, device=None, dataset=DATASET, voices_dir=None, checkpoint=CHECKPOINT):
        import torch
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.checkpoint = checkpoint
        self._voices = load_voices(dataset, local_dir=voices_dir)
        self._model = None
        self._lock = threading.Lock()

    # ---- voices ---------------------------------------------------------------
    def voices(self):
        """All reference voices: {name: Voice}."""
        return dict(self._voices)

    def voice(self, name):
        return find_voice(self._voices, name)

    # ---- model ----------------------------------------------------------------
    def load(self):
        if self._model is None:
            import torch
            from huggingface_hub import snapshot_download
            _patch_torchaudio_load()
            from omnivoice.utils import lang_map
            for code in LANGUAGES.values():
                lang_map.LANG_IDS.add(code)
            from omnivoice import OmniVoice
            local = snapshot_download(MODEL_REPO, allow_patterns=[f"{self.checkpoint}/*"],
                                      ignore_patterns=["*optimizer*", "*random_states*", "*scheduler*", "*train_config*"])
            gpu = self.device.startswith("cuda")
            if gpu:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            self._model = OmniVoice.from_pretrained(os.path.join(local, self.checkpoint), device_map=self.device,
                                                    dtype=torch.float16 if gpu else torch.float32)
        return self._model

    # ---- speech ---------------------------------------------------------------
    def speak(self, text, voice=None, language="twi-asante", ref_audio=None, ref_text=None,
              guidance=2.0, steps=32, speed=None, out=None):
        """Speak `text` in a labelled `voice` (or your own `ref_audio` + optional `ref_text`).

        Returns (sample_rate, float32 samples); also writes a WAV file if `out` is given.
        """
        import soundfile as sf
        from omnivoice import OmniVoiceGenerationConfig
        text = " ".join((text or "").split())
        if not text:
            raise ValueError("nothing to say: text is empty")
        if ref_audio is None:
            v = self.voice(voice) if voice else next(iter(self._voices.values()))
            ref_audio, ref_text = v.audio, v.text
        model = self.load()
        kw = {}
        if speed and float(speed) != 1.0:
            kw["speed"] = float(speed)
        with self._lock:  # one generation at a time on the model
            prompt = model.create_voice_clone_prompt(ref_audio=ref_audio, ref_text=ref_text)
            audio = model.generate(text=text, language=language_code(language), voice_clone_prompt=prompt,
                                   generation_config=OmniVoiceGenerationConfig(num_step=int(steps),
                                                                               guidance_scale=float(guidance),
                                                                               denoise=True), **kw)
        wav = trim_silence(np.asarray(audio[0], dtype=np.float32).reshape(-1), SAMPLE_RATE)
        if out:
            sf.write(out, wav, SAMPLE_RATE, subtype="PCM_16")
        return SAMPLE_RATE, wav

    def speak_wav_bytes(self, *args, **kwargs):
        """Like speak(), but returns the audio as WAV file bytes (for APIs)."""
        import soundfile as sf
        sr, wav = self.speak(*args, **kwargs)
        buf = io.BytesIO()
        sf.write(buf, wav, sr, format="WAV", subtype="PCM_16")
        return buf.getvalue()
