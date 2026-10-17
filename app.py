import os
import sys

import numpy as np
import soundfile as sf
import torch
import torchaudio
import gradio as gr
from huggingface_hub import snapshot_download

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def _sf_load(path, *args, **kwargs):
    """torchaudio.load via soundfile (avoids torchcodec / CUDA-version coupling)."""
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    return torch.from_numpy(data.T.copy()), sr


torchaudio.load = _sf_load

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
from prompts import DATASET, load_choices  # noqa: E402

VOICES = load_choices()  # "label" / "label #n" -> reference clip from ghanaopenai/ghana-audio-prompts

# language label -> OmniVoice language code (languages the omnivoice-ghana fine-tune was trained on)
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

EXAMPLES = [
    ["Gyae me kwan na mɛtwa wo ti! Obiara ntumi nsi me kwan wɔ ha!", "Twi (Asante)"],
    ["Nsuro, me nua. Yɛbɛdi nkunim bɔɔbɔ.", "Twi (Asante)"],
    ["Me daase pii, Nyame nhyira wo.", "Twi (Asante)"],
    ["Akpe na wò, Mawu neyra wò.", "Ewe"],
    ["Bi zɔri mi ni be chɛŋ shukuru be konko.", "Dagbani"],
]

_model = {}


def load_model():
    if "m" not in _model:
        from omnivoice.utils import lang_map
        for code in LANGUAGES.values():
            lang_map.LANG_IDS.add(code)
        from omnivoice import OmniVoice
        local = snapshot_download(
            "ghananlpcommunity/omnivoice-ghana",
            allow_patterns=["checkpoint-45000/*"],
            ignore_patterns=["*optimizer*", "*random_states*", "*scheduler*", "*train_config*"],
        )
        _model["m"] = OmniVoice.from_pretrained(
            os.path.join(local, "checkpoint-45000"), device_map="cuda:0", dtype=torch.float16
        )
    return _model["m"]


def trim_silence(wav, sr, floor_db=-40.0, pad=0.15):
    """Cut leading/trailing silence (frames quieter than floor_db below peak)."""
    n = int(0.02 * sr)
    frames = wav[: len(wav) // n * n].reshape(-1, n)
    db = 20 * np.log10(np.sqrt((frames ** 2).mean(1)) + 1e-9)
    loud = np.where(db > db.max() + floor_db)[0]
    if len(loud) == 0:
        return wav
    p = int(pad * sr)
    return wav[max(0, loud[0] * n - p): min(len(wav), (loud[-1] + 1) * n + p)]


def preview(name):
    """Reference clip + its transcription, shown when a voice is chosen."""
    clip = VOICES.get(name)
    return (clip["audio"], f"“{clip['text']}”") if clip else (None, "")


def synthesize(text, language, voice, own_audio, own_transcript, cfg, steps):
    if not text or not text.strip():
        raise gr.Error("Type what the character says first.")
    text = " ".join(text.split())

    if own_audio:
        ref_wav, ref_text = own_audio, (own_transcript or "").strip() or None
        info = "**Reference voice:** your uploaded clip"
    else:
        clip = VOICES[voice]
        ref_wav, ref_text = clip["audio"], clip["text"]
        info = f"**Reference voice:** {voice}"

    from omnivoice import OmniVoiceGenerationConfig
    model = load_model()
    prompt = model.create_voice_clone_prompt(ref_audio=ref_wav, ref_text=ref_text)
    audio = model.generate(
        text=text, language=LANGUAGES[language], voice_clone_prompt=prompt,
        generation_config=OmniVoiceGenerationConfig(num_step=int(steps), guidance_scale=float(cfg), denoise=True),
    )
    wav = trim_silence(np.asarray(audio[0], dtype=np.float32).reshape(-1), 24000)
    return (24000, wav), info


with gr.Blocks(title="Ghana OmniVoice TTS") as demo:
    gr.Markdown(
        "# 🇬🇭 Ghana OmniVoice TTS\n"
        "Type a line a game character would say, pick the language and a **reference voice**, and press **Generate**.  \n"
        f"The voices are [{len(VOICES)} hand-checked, labelled Twi clips](https://huggingface.co/datasets/{DATASET}); "
        "the model speaks your line in that voice and mood."
    )
    with gr.Row():
        with gr.Column():
            text = gr.Textbox(label="What does the character say?", lines=3,
                              placeholder="e.g. Fi me dan mu seesei ara!")
            language = gr.Dropdown(list(LANGUAGES), value="Twi (Asante)", label="Language")
            first = next(iter(VOICES))
            voice = gr.Dropdown(list(VOICES), value=first, label="Reference voice (voice - emotion)")
            ref_audio = gr.Audio(value=preview(first)[0], label="Reference clip", type="filepath", interactive=False)
            ref_text = gr.Markdown(preview(first)[1])
            btn = gr.Button("Generate", variant="primary")
            with gr.Accordion("Advanced", open=False):
                own_audio = gr.Audio(label="Use your own reference voice instead (3–10 s)", type="filepath")
                own_transcript = gr.Textbox(label="Transcript of your reference clip (optional, improves cloning)")
                cfg = gr.Slider(1.0, 4.0, value=2.0, step=0.1, label="Guidance (CFG)")
                steps = gr.Slider(8, 64, value=32, step=2, label="Steps")
        with gr.Column():
            out = gr.Audio(label="Generated speech", type="numpy")
            ref_info = gr.Markdown()

    gr.Examples(EXAMPLES, inputs=[text, language])
    gr.Markdown("The first generation takes a minute or two while the model loads. Model: "
                "[ghananlpcommunity/omnivoice-ghana](https://huggingface.co/ghananlpcommunity/omnivoice-ghana) "
                "(checkpoint-45000).")

    voice.change(preview, voice, [ref_audio, ref_text])
    btn.click(synthesize, [text, language, voice, own_audio, own_transcript, cfg, steps], [out, ref_info])

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(), show_error=True)
