import os
import subprocess
import pandas as pd
from pathlib import Path
from pathlib import Path
import tgt
from pydub import AudioSegment



def align_audio_dataframe(
    df: pd.DataFrame,
    audio_col: str,
    transcript_col: str,
    audio_dir: str,
    output_dir: str = "output_textgrid",
    acoustic_model: str = "french_mfa",
    dictionary: str = "french_mfa",
    jobs: int = 4,
) -> None:
    audio_dir, output_dir = Path(audio_dir), Path(output_dir)
    mfa_input = Path("mfa_input_tmp")

    mfa_input.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    for _, row in df.iterrows():
        src = audio_dir / row[audio_col]
        dst = mfa_input / row[audio_col]
        dst.write_bytes(src.read_bytes())
        (mfa_input / f"{src.stem}.lab").write_text(str(row[transcript_col]).strip(), encoding="utf-8")

    subprocess.run(
        ["mfa", "align", str(mfa_input), dictionary, acoustic_model, str(output_dir), "--jobs", str(jobs), "--clean"],
        check=True,
    )

df = ""
audio_dir = ""
exemple = align_audio_dataframe(df, "audio_file", "traduction", audio_dir)



#Extract phoneme 
def extract_phoneme(audio_path: str, textgrid_path: str, phoneme: str) -> None:
    tg = tgt.io.read_textgrid(textgrid_path)
    tier = tg.get_tier_by_name("phones")

    intervals = [i for i in tier.intervals if i.text == phoneme]
    if not intervals:
        raise ValueError(f"Phonème '{phoneme}' introuvable dans {textgrid_path}")

    audio = AudioSegment.from_file(audio_path)
    stem = Path(audio_path).stem

    for idx, interval in enumerate(intervals):
        start_ms = interval.start_time * 1000
        end_ms = interval.end_time * 1000
        segment = audio[start_ms:end_ms]

        suffix = f"_{idx}" if len(intervals) > 1 else ""
        out_path = Path(audio_path).parent / f"{stem}_{phoneme}{suffix}.wav"
        segment.export(out_path, format="wav")
        print(f"Saved: {out_path}")


#Extract audio
def extract_word(audio_path: str, textgrid_path: str, word: str) -> None:
    tg = tgt.io.read_textgrid(textgrid_path)
    tier = tg.get_tier_by_name("words")

    intervals = [i for i in tier.intervals if i.text == word]
    if not intervals:
        raise ValueError(f"Mot '{word}' introuvable dans {textgrid_path}")

    audio = AudioSegment.from_file(audio_path)
    stem = Path(audio_path).stem

    for idx, interval in enumerate(intervals):
        segment = audio[interval.start_time * 1000 : interval.end_time * 1000]
        suffix = f"_{idx}" if len(intervals) > 1 else ""
        out_path = Path(audio_path).parent / f"{stem}_{word}{suffix}.wav"
        segment.export(out_path, format="wav")
        print(f"Saved: {out_path}")