---
title: Ghana OmniVoice TTS
emoji: 🎙️
colorFrom: yellow
colorTo: red
sdk: gradio
sdk_version: 6.30.0
app_file: app.py
pinned: false
license: mit
short_description: Game-character voices for Ghanaian languages
---

# Ghana OmniVoice TTS

Type a line a game character would say, pick the language and a reference voice, and press **Generate**.

- **Model:** [OmniVoice-Ghana](https://huggingface.co/ghananlpcommunity/omnivoice-ghana) (checkpoint-45000),
  a fine-tune of OmniVoice on Ghanaian languages.
- **Reference voice:** chosen by label (voice - emotion, e.g. `adult male - angry`) from the
  hand-checked Twi clips in [ghanaopenai/ghana-audio-prompts](https://huggingface.co/datasets/ghanaopenai/ghana-audio-prompts)
  (curated with [audio2prompt](https://github.com/GhanaOpenAI/audio2prompt)). Labels shared by several
  clips appear as `label #1`, `#2`, ... The clip plays when you pick it, and its checked transcription
  is passed to the model as the reference text. New clips pushed to the dataset appear after a Space restart.
- **Your own voice:** under Advanced, upload a 3–10 s clip (and optionally its transcript) instead.

Code: [github.com/GhanaOpenAI/ghana-omnivoice-tts](https://github.com/GhanaOpenAI/ghana-omnivoice-tts)
