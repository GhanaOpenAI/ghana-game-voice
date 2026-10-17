"""Reference voices for the Space: the labelled clips in the ghanaopenai/ghana-audio-prompts dataset.

Each clip has a checked Twi transcription and a label "voice - emotion". The user picks a label; when
several clips share one, they are listed as "label #1", "label #2", ...
"""
import csv
import os
from collections import Counter, defaultdict

from huggingface_hub import snapshot_download

DATASET = "ghanaopenai/ghana-audio-prompts"


def load_choices():
    """-> {dropdown name: {"audio", "text", "label", "duration"}}, sorted by label."""
    # Download into the app folder: Gradio only serves files from there (or /tmp), not from the HF cache.
    root = snapshot_download(DATASET, repo_type="dataset",
                             local_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompt_audio"))
    with open(os.path.join(root, "metadata.csv"), encoding="utf-8") as f:
        rows = sorted(csv.DictReader(f), key=lambda r: (r["label"], r["file_name"]))
    counts, seen = Counter(r["label"] for r in rows), defaultdict(int)
    choices = {}
    for r in rows:
        seen[r["label"]] += 1
        name = r["label"] if counts[r["label"]] == 1 else f"{r['label']} #{seen[r['label']]}"
        choices[name] = {"audio": os.path.join(root, r["file_name"]), "text": r["transcription"].strip(),
                         "label": r["label"], "duration": float(r.get("duration_sec") or 0)}
    return choices
