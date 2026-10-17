"""HTTP API (FastAPI). Run with `ghana-game-voice serve`; interactive docs at /docs."""
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

from .tts import LANGUAGES, GameVoice


class SpeakRequest(BaseModel):
    text: str = Field(..., examples=["Fi me dan mu seesei ara!"])
    voice: Optional[str] = Field(None, description="voice name from GET /voices (default: the first)",
                                 examples=["threat"])
    language: str = Field("twi-asante", description="language code or name, see GET /languages")
    guidance: float = 2.0
    steps: int = 32
    speed: Optional[float] = None


def create_app(gv: Optional[GameVoice] = None, preload=False) -> FastAPI:
    gv = gv or GameVoice()
    if preload:
        gv.load()
    app = FastAPI(title="Ghana Game Voice",
                  description="Speak lines in Ghanaian languages with OmniVoice-Ghana in labelled reference voices.")

    @app.get("/voices")
    def voices():
        """All reference voices (name, label, speaker, emotion, transcription, duration)."""
        return [v.to_dict() for v in gv.voices().values()]

    @app.get("/voices/{name}/audio", response_class=FileResponse)
    def voice_audio(name: str):
        """The reference clip for a voice (WAV)."""
        try:
            return FileResponse(gv.voice(name).audio, media_type="audio/wav")
        except KeyError as e:
            raise HTTPException(404, e.args[0])

    @app.get("/languages")
    def languages():
        return LANGUAGES

    @app.post("/speak", responses={200: {"content": {"audio/wav": {}}}})
    def speak(req: SpeakRequest):
        """Generate speech; returns a WAV file."""
        try:
            wav = gv.speak_wav_bytes(req.text, voice=req.voice, language=req.language, guidance=req.guidance,
                                     steps=req.steps, speed=req.speed)
        except (KeyError, ValueError) as e:
            raise HTTPException(400, e.args[0])
        return Response(wav, media_type="audio/wav")

    @app.get("/health")
    def health():
        return {"ok": True, "model_loaded": gv._model is not None, "device": gv.device, "voices": len(gv.voices())}

    return app
