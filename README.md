# Ghana Game Voice

Speak lines in Ghanaian languages with [OmniVoice-Ghana](https://huggingface.co/ghananlpcommunity/omnivoice-ghana),
in a **labelled reference voice**: pick `adult male - angry`, `adult female - tender`, `young male - happy`, ...
from the hand-checked clips in [ghanaopenai/ghana-audio-prompts](https://huggingface.co/datasets/ghanaopenai/ghana-audio-prompts),
and the model says your line in that voice and mood. Use it from **Python**, the **command line** or an **HTTP API**.

Try it in the browser: [Ghana Game Voice Space](https://huggingface.co/spaces/ghananlpcommunity/ghana-game-voice).

## Install

A GPU is strongly recommended (CPU works, slowly). Install the PyTorch build that matches your CUDA
driver first, e.g. for CUDA 12.8:

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install "ghana-game-voice[api] @ git+https://github.com/GhanaOpenAI/ghana-game-voice.git"
```

(`[api]` adds the HTTP server; leave it out if you only need Python and the CLI.) The model
(~2 GB) and the reference voices download from Hugging Face on first use.

## Voices

Every clip in the dataset has a label `voice - emotion`. Labels shared by several clips are numbered
`adult male - angry #1`, `#2`, ...; asking for the bare label gives `#1`.

```bash
ghana-game-voice voices            # name, length and transcription of every voice
ghana-game-voice voices --json
ghana-game-voice languages         # twi-asante, twi-akuapem, fat, ewe, dag, dga, ada, nzi, hau, en
```

## Command line

```bash
ghana-game-voice speak "Fi me dan mu seesei ara!" -v "adult male - angry" -l twi -o line.wav
ghana-game-voice speak "Akpe na wò, Mawu neyra wò." -v "adult female - grateful" -l ewe -o ewe.wav
ghana-game-voice speak "Me daase pii." --ref-audio my_voice.wav --ref-text "…" -o mine.wav   # your own voice
```

Options: `--guidance` (2.0), `--steps` (32), `--speed`, `--device cuda:0|cpu`.

## Python

```python
from ghana_game_voice import GameVoice

gv = GameVoice()                                   # GPU if available
for v in gv.voices().values():
    print(v.name, v.speaker, v.emotion, v.text)

sr, wav = gv.speak("Fi me dan mu seesei ara!", voice="adult male - angry", language="twi",
                   out="line.wav")                 # returns (24000, float32 samples), and writes the file
wav_bytes = gv.speak_wav_bytes("Me daase pii.", voice="adult female - grateful")
```

The model loads on the first `speak()` and stays loaded; later lines take about a second or two on a GPU.

## HTTP API

```bash
ghana-game-voice serve --port 8000 --preload       # interactive docs at http://localhost:8000/docs
```

| endpoint | |
|---|---|
| `GET /voices` | all voices: name, label, speaker, emotion, transcription, duration |
| `GET /voices/{name}/audio` | a voice's reference clip (WAV) |
| `GET /languages` | supported languages |
| `POST /speak` | `{"text", "voice", "language", "guidance", "steps", "speed"}` → WAV audio |
| `GET /health` | status |

```bash
curl -X POST localhost:8000/speak -H "Content-Type: application/json" \
     -d '{"text": "Wo yɛ aboa!", "voice": "adult male - angry #2", "language": "twi"}' -o line.wav
```

## Languages

Asante and Akuapem Twi, Fante, Ewe, Dagbani, Dagaare, Dangme, Nzema, Hausa and English (codes from
`ghana-game-voice languages`; names like `Ewe` or `twi` work too). The reference voices are mostly
Twi speakers; they can speak any of these languages.

## Repo layout

- `ghana_game_voice/`: the package (`voices.py` loads the labelled voices, `tts.py` wraps OmniVoice,
  `cli.py`, `api.py`).
- `space/`: the Hugging Face Space, a Gradio page over the package.

## Credits

Model: [ghananlpcommunity/omnivoice-ghana](https://huggingface.co/ghananlpcommunity/omnivoice-ghana), built on
[OmniVoice](https://github.com/k2-fsa/OmniVoice). Voices: [ghanaopenai/ghana-audio-prompts](https://huggingface.co/datasets/ghanaopenai/ghana-audio-prompts),
curated with [audio2prompt](https://github.com/GhanaOpenAI/audio2prompt).
