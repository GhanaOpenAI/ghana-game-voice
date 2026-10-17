"""Hugging Face Space: a web page over the ghana-game-voice package."""
import os

import gradio as gr

from ghana_game_voice import DATASET, LANGUAGES, GameVoice

# Voices are downloaded into the app folder: Gradio only serves files from there (or /tmp).
GV = GameVoice(voices_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompt_audio"))
VOICES = GV.voices()

# line, language, voice: each example sets all three
EXAMPLES = [
    ["Fi me kwan so! Gyae me kwan na mɛtwa wo ti!", "Twi (Asante)", "threat"],
    ["Worenguan mfi me nkyɛn!", "Twi (Asante)", "cold threat"],
    ["Tow hyɛ wɔn so mprempren!", "Twi (Asante)", "enraged fighter (male)"],
    ["Mesrɛ wo, hu me mmɔbɔ!", "Twi (Asante)", "begging for life"],
    ["Dwane! Wɔreba!", "Twi (Asante)", "terrified"],
    ["Do le nye mɔ dzi!", "Ewe", "get out!"],
    ["Màsi le asinye o!", "Ewe", "cold threat"],
    ["Meɖe kuku, kpɔ nublanui nam!", "Ewe", "pleading (female)"],
]
EXAMPLES = [e for e in EXAMPLES if e[2] in VOICES]


def preview(name):
    """Reference clip + its transcription, shown when a voice is chosen."""
    v = VOICES.get(name)
    return (v.audio, f"{v.label} · “{v.text}”") if v else (None, "")


def synthesize(text, language, voice, own_audio, own_transcript, guidance, steps):
    if not text or not text.strip():
        raise gr.Error("Type what the character says first.")
    if own_audio:
        sr, wav = GV.speak(text, language=language, ref_audio=own_audio,
                           ref_text=(own_transcript or "").strip() or None, guidance=guidance, steps=steps)
        return (sr, wav), "**Reference voice:** your uploaded clip"
    sr, wav = GV.speak(text, voice=voice, language=language, guidance=guidance, steps=steps)
    return (sr, wav), f"**Reference voice:** {voice}"


with gr.Blocks(title="Ghana Game Voice") as demo:
    gr.Markdown(
        "# 🇬🇭 Ghana Game Voice\n"
        "Type a line a character says in battle, in **Twi** or **Ewe**, pick a **reference voice**, and press **Generate**.  \n"
        f"The {len(VOICES)} voices are combat-ready clips (angry, threatening, taunting, pleading, scared, ...) from "
        f"[hand-checked, labelled recordings](https://huggingface.co/datasets/{DATASET}); the model speaks your line "
        "in that voice and mood. Also available as a Python package, CLI and API: "
        "[github.com/GhanaOpenAI/ghana-game-voice](https://github.com/GhanaOpenAI/ghana-game-voice)."
    )
    with gr.Row():
        with gr.Column():
            text = gr.Textbox(label="What does the character say?", lines=3, placeholder="e.g. Fi me kwan so!")
            language = gr.Dropdown(list(LANGUAGES), value="Twi (Asante)", label="Language")
            first = next(iter(VOICES))
            voice = gr.Dropdown(list(VOICES), value=first, label="Voice")
            ref_audio = gr.Audio(value=preview(first)[0], label="Reference clip", type="filepath", interactive=False)
            ref_text = gr.Markdown(preview(first)[1])
            btn = gr.Button("Generate", variant="primary")
            with gr.Accordion("Advanced", open=False):
                own_audio = gr.Audio(label="Use your own reference voice instead (3–10 s)", type="filepath")
                own_transcript = gr.Textbox(label="Transcript of your reference clip (optional, improves cloning)")
                guidance = gr.Slider(1.0, 4.0, value=2.0, step=0.1, label="Guidance (CFG)")
                steps = gr.Slider(8, 64, value=32, step=2, label="Steps")
        with gr.Column():
            out = gr.Audio(label="Generated speech", type="numpy")
            ref_info = gr.Markdown()

    gr.Examples(EXAMPLES, inputs=[text, language, voice])
    gr.Markdown("The first generation takes a minute or two while the model loads. Model: "
                "[ghananlpcommunity/omnivoice-ghana](https://huggingface.co/ghananlpcommunity/omnivoice-ghana) "
                "(checkpoint-45000).")

    voice.change(preview, voice, [ref_audio, ref_text])
    btn.click(synthesize, [text, language, voice, own_audio, own_transcript, guidance, steps], [out, ref_info])

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(), show_error=True)
