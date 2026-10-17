"""Labelled reference voices from the ghanaopenai/ghana-audio-prompts dataset.

Each clip has a checked Twi transcription and a label "voice - emotion" (e.g. "adult male - angry").
When several clips share a label they are named "label #1", "label #2", ...; asking for the bare
label gives #1.
"""
import csv
import os
from collections import Counter, defaultdict
from dataclasses import dataclass

DATASET = "ghanaopenai/ghana-audio-prompts"


@dataclass
class Voice:
    name: str          # unique name, e.g. "adult male - angry #2"
    label: str         # e.g. "adult male - angry"
    speaker: str       # voice characteristic, e.g. "adult male"
    emotion: str       # e.g. "angry"
    audio: str         # path to the reference clip
    text: str          # its checked transcription
    duration: float

    def to_dict(self):
        return {k: getattr(self, k) for k in ("name", "label", "speaker", "emotion", "text", "duration")}


def load_voices(dataset=DATASET, local_dir=None, revision=None):
    """Download the dataset (cached) and return {name: Voice}, sorted by label."""
    from huggingface_hub import snapshot_download
    root = snapshot_download(dataset, repo_type="dataset", local_dir=local_dir, revision=revision)
    with open(os.path.join(root, "metadata.csv"), encoding="utf-8") as f:
        rows = sorted(csv.DictReader(f), key=lambda r: (r["label"], r["file_name"]))
    counts, seen = Counter(r["label"] for r in rows), defaultdict(int)
    voices = {}
    for r in rows:
        label = r["label"].strip()
        seen[label] += 1
        name = label if counts[label] == 1 else f"{label} #{seen[label]}"
        speaker, _, emotion = label.partition(" - ")
        voices[name] = Voice(name=name, label=label, speaker=speaker.strip(), emotion=emotion.strip(),
                             audio=os.path.join(root, r["file_name"]), text=r["transcription"].strip(),
                             duration=float(r.get("duration_sec") or 0))
    return voices


def find_voice(voices, name):
    """Exact name, or a bare label (-> its #1), case-insensitive."""
    if name in voices:
        return voices[name]
    low = {k.lower(): v for k, v in voices.items()}
    key = name.strip().lower()
    if key in low:
        return low[key]
    if f"{key} #1" in low:
        return low[f"{key} #1"]
    raise KeyError(f"unknown voice {name!r}; see `ghana-game-voice voices`")
