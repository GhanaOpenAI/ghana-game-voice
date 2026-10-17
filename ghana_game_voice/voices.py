"""Labelled reference voices from the ghanaopenai/ghana-audio-prompts dataset.

Each clip has a checked transcription and a label "voice - emotion" (e.g. "adult male - angry"). By default
only the "combat" set is used: the clips picked for a combat game, each with a game role.
When several clips share a label they are named "label #1", "label #2", ...; asking for the bare
label gives #1.
"""
import csv
import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass

DATASET = "ghanaopenai/ghana-audio-prompts"
# Built-in voice sets: "combat" = the clips picked for a combat game (combat.json); "all" = every clip.
COMBAT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "combat.json")


@dataclass
class Voice:
    name: str          # unique name, e.g. "adult male - angry #2"
    label: str         # e.g. "adult male - angry"
    speaker: str       # voice characteristic, e.g. "adult male"
    emotion: str       # e.g. "angry"
    audio: str         # path to the reference clip
    text: str          # its checked transcription
    duration: float
    file: str = ""     # path inside the dataset, e.g. "audio/0168.wav"
    role: str = ""     # game role in a curated set, e.g. "taunt"

    def to_dict(self):
        return {k: getattr(self, k) for k in ("name", "label", "speaker", "emotion", "role", "text", "duration", "file")}


def load_voices(dataset=DATASET, local_dir=None, revision=None, voice_set="combat"):
    """Download the dataset (cached) and return {name: Voice}, sorted by label.

    voice_set: "combat" (default: the clips picked for a combat game), "all", or a path to a JSON list of
    {"file": "audio/xxxx.wav", "role": "..."} entries to make your own set.
    """
    from huggingface_hub import snapshot_download
    root = snapshot_download(dataset, repo_type="dataset", local_dir=local_dir, revision=revision)
    with open(os.path.join(root, "metadata.csv"), encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    roles = {}
    if voice_set != "all":
        with open(COMBAT if voice_set == "combat" else voice_set, encoding="utf-8") as f:
            roles = {e["file"]: e.get("role", "") for e in json.load(f)}
        rows = [r for r in rows if r["file_name"] in roles]
    rows.sort(key=lambda r: (r["label"], r["file_name"]))
    counts, seen = Counter(r["label"] for r in rows), defaultdict(int)
    voices = {}
    for r in rows:
        label = r["label"].strip()
        seen[label] += 1
        name = label if counts[label] == 1 else f"{label} #{seen[label]}"
        speaker, _, emotion = label.partition(" - ")
        voices[name] = Voice(name=name, label=label, speaker=speaker.strip(), emotion=emotion.strip(),
                             audio=os.path.join(root, r["file_name"]), text=r["transcription"].strip(),
                             duration=float(r.get("duration_sec") or 0), file=r["file_name"],
                             role=roles.get(r["file_name"], ""))
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
