# Ghana Game Voice

Voice lines for a combat game in **Asante Twi** and **Ewe**, spoken by
[OmniVoice-Ghana](https://huggingface.co/ghananlpcommunity/omnivoice-ghana) in a **labelled reference voice**:
pick an enraged fighter, a cold threat, a taunt, a begging enemy, a terrified villager, ... from hand-checked
clips in [ghanaopenai/ghana-audio-prompts](https://huggingface.co/datasets/ghanaopenai/ghana-audio-prompts),
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

By default you get the **combat set**: 19 short, expressive clips picked for a combat game, each with a
label `voice - emotion` and a game role:

| roles | voices |
|---|---|
| enraged fighter, confrontation, boss rant, threat, "get out!" | `adult male - angry #1`-`#5`, `adult female - angry` |
| cold, quiet threat · last warning · seething | `adult male - calm`, `adult male - warning`, `adult male - frustrated` |
| taunt · mocking | `adult male - insulting`, `adult female - sarcastic`, `adult male - sarcastic` |
| suspicious guard · warning / mission giver | `adult male - doubtful`, `young female - stern` |
| begging for life · desperate plea · pleading | `adult male - pleading`, `adult male - disagreeing`, `young female - pleading` |
| terrified · wounded | `young female - anxiety`, `adult female - hurt` |

Labels shared by several clips are numbered `#1`, `#2`, ...; asking for the bare label gives `#1`.
The voices are Twi speakers; they speak Ewe lines too.

```bash
ghana-game-voice voices              # name, role, length and transcription of every voice
ghana-game-voice voices --json
ghana-game-voice --all-voices voices # every clip in the dataset, not just the combat set
ghana-game-voice languages           # twi-asante, ewe
```

Make your own set: a JSON list of `{"file": "audio/0168.wav", "role": "..."}` entries, passed as
`--voice-set my_set.json` (CLI) or `GameVoice(voice_set="my_set.json")`; see `ghana_game_voice/combat.json`.

## Command line

```bash
ghana-game-voice speak "Fi me kwan so!" -v "adult male - angry #4" -l twi -o line.wav
ghana-game-voice speak "Màsi le asinye o!" -v "adult male - calm" -l ewe -o ewe.wav
ghana-game-voice speak "Me daase pii." --ref-audio my_voice.wav --ref-text "…" -o mine.wav   # your own voice
```

Options: `--guidance` (2.0), `--steps` (32), `--speed`, `--device cuda:0|cpu`.

## Python

```python
from ghana_game_voice import GameVoice

gv = GameVoice()                                   # GPU if available
for v in gv.voices().values():
    print(v.name, v.role, v.text)

sr, wav = gv.speak("Fi me kwan so!", voice="adult male - angry #4", language="twi",
                   out="line.wav")                 # returns (24000, float32 samples), and writes the file
wav_bytes = gv.speak_wav_bytes("Meɖe kuku, kpɔ nublanui nam!", voice="young female - pleading", language="ewe")
```

The model loads on the first `speak()` and stays loaded; later lines take about a second or two on a GPU.

## HTTP API

```bash
ghana-game-voice serve --port 8000 --preload       # interactive docs at http://localhost:8000/docs
```

| endpoint | |
|---|---|
| `GET /voices` | all voices: name, label, speaker, emotion, role, transcription, duration |
| `GET /voices/{name}/audio` | a voice's reference clip (WAV) |
| `GET /languages` | supported languages |
| `POST /speak` | `{"text", "voice", "language", "guidance", "steps", "speed"}` → WAV audio |
| `GET /health` | status |

```bash
curl -X POST localhost:8000/speak -H "Content-Type: application/json" \
     -d '{"text": "Wo yɛ aboa!", "voice": "adult male - angry #2", "language": "twi"}' -o line.wav
```

## Languages

**Asante Twi** (`twi-asante`, or `twi`) and **Ewe** (`ewe`).

## Repo layout

- `ghana_game_voice/`: the package (`voices.py` loads the labelled voices, `tts.py` wraps OmniVoice,
  `cli.py`, `api.py`).
- `space/`: the Hugging Face Space, a Gradio page over the package.

## Credits

Model: [ghananlpcommunity/omnivoice-ghana](https://huggingface.co/ghananlpcommunity/omnivoice-ghana), built on
[OmniVoice](https://github.com/k2-fsa/OmniVoice). Voices: [ghanaopenai/ghana-audio-prompts](https://huggingface.co/datasets/ghanaopenai/ghana-audio-prompts),
curated with [audio2prompt](https://github.com/GhanaOpenAI/audio2prompt).
