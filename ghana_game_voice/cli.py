"""Command line: `ghana-game-voice voices | speak | serve`."""
import argparse
import json
import sys


def main(argv=None):
    ap = argparse.ArgumentParser(prog="ghana-game-voice",
                                 description="Speak lines in Ghanaian languages with OmniVoice-Ghana in labelled reference voices.")
    ap.add_argument("--all-voices", action="store_true",
                    help="use every clip in the dataset instead of the combat set")
    ap.add_argument("--voice-set", default=None, help="a JSON file of {file, role} entries: your own voice set")
    sub = ap.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("voices", help="list the reference voices")
    v.add_argument("--json", action="store_true", help="output JSON")

    s = sub.add_parser("speak", help="generate speech to a WAV file")
    s.add_argument("text", help="what the character says")
    s.add_argument("-v", "--voice", help='voice name or label, e.g. "adult male - angry" or "adult female - stern #2"')
    s.add_argument("-l", "--language", default="twi-asante", help="twi-asante (default) or ewe")
    s.add_argument("-o", "--out", default="speech.wav", help="output WAV file (default: speech.wav)")
    s.add_argument("--ref-audio", help="use your own 3-10 s reference clip instead of a labelled voice")
    s.add_argument("--ref-text", help="transcription of --ref-audio (optional, improves cloning)")
    s.add_argument("--guidance", type=float, default=2.0)
    s.add_argument("--steps", type=int, default=32)
    s.add_argument("--speed", type=float, default=None)
    s.add_argument("--device", default=None, help="cuda:0 / cpu (default: GPU if available)")

    sv = sub.add_parser("serve", help="run the HTTP API (docs at /docs)")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8000)
    sv.add_argument("--device", default=None)
    sv.add_argument("--preload", action="store_true", help="load the model at startup instead of on first request")

    l = sub.add_parser("languages", help="list supported languages")  # noqa: E741

    a = ap.parse_args(argv)
    voice_set = "all" if a.all_voices else (a.voice_set or "combat")

    if a.cmd == "languages":
        from .tts import LANGUAGES
        for name, code in LANGUAGES.items():
            print(f"{code:12s} {name}")
        return

    if a.cmd == "voices":
        from .voices import load_voices
        vs = load_voices(voice_set=voice_set)
        if a.json:
            print(json.dumps([x.to_dict() for x in vs.values()], ensure_ascii=False, indent=1))
        else:
            width, rw = max(len(n) for n in vs), max(len(x.role) for x in vs.values())
            for x in vs.values():
                print(f"{x.name:{width}s}  {x.role:{rw}s}  {x.duration:4.1f}s  {x.text}")
        return

    if a.cmd == "speak":
        from .tts import GameVoice
        gv = GameVoice(device=a.device, voice_set=voice_set)
        try:
            gv.speak(a.text, voice=a.voice, language=a.language, ref_audio=a.ref_audio, ref_text=a.ref_text,
                     guidance=a.guidance, steps=a.steps, speed=a.speed, out=a.out)
        except (KeyError, ValueError) as e:
            sys.exit(e.args[0])
        print(a.out)
        return

    if a.cmd == "serve":
        try:
            import uvicorn
            from .api import create_app
        except ImportError:
            sys.exit('the API needs extra packages: pip install "ghana-game-voice[api]"')
        from .tts import GameVoice
        uvicorn.run(create_app(GameVoice(device=a.device, voice_set=voice_set), preload=a.preload), host=a.host, port=a.port)


if __name__ == "__main__":
    main()
