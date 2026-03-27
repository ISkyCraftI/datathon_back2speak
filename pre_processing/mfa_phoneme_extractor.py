"""
MFA Phoneme Extractor — Français
=================================
Pipeline complet :
  1. Lit un dossier input/ contenant :
       input/
           audio/          <- fichiers .wav
           transcriptions/ <- fichiers .txt (même nom que les .wav)
  2. Aligne avec Montreal Forced Aligner (MFA) — modèles français
  3. Génère un TextGrid par fichier audio
  4. Extrait un phonème cible depuis chaque TextGrid (ex: ʃ pour "ch")
  5. Exporte les segments audio en .wav

Prérequis :
  pip install praatio soundfile numpy
  conda install -c conda-forge montreal-forced-aligner
  mfa model download acoustic french_mfa
  mfa model download dictionary french_mfa

Phonèmes "ch" en français (dictionnaire french_mfa) :
  ʃ  ->  "ch" consonantique  (chat, chose, chef, choux...)
"""

import os
import sys
import shutil
import subprocess
import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
from praatio import textgrid as tgio


# -------------------------------------------------
# CONSTANTES
# -------------------------------------------------

DEFAULT_ACOUSTIC_MODEL = "french_mfa"
DEFAULT_DICTIONARY     = "french_mfa"
CH_PHONEME             = "ʃ"   # phonème "ch" en français (IPA)


# -------------------------------------------------
# 1.  ALIGNEMENT MFA
# -------------------------------------------------

def run_mfa_alignment(
    input_dir: str,
    output_dir: str,
    acoustic_model: str = DEFAULT_ACOUSTIC_MODEL,
    dictionary: str = DEFAULT_DICTIONARY,
) -> Path:
    """
    Prepare le corpus MFA depuis la structure suivante et lance l'alignement :

        input_dir/
            audio/              <- fichiers .wav
            transcriptions/     <- fichiers .txt (meme nom que les .wav)

    MFA requiert que chaque audio et sa transcription soient dans le meme
    dossier avec la meme racine. Cette fonction construit un corpus temporaire
    en copiant les paires (audio.wav / audio.lab) cote a cote.

        corpus_tmp/
            speaker1/
                fichier1.wav
                fichier1.lab
                fichier2.wav
                fichier2.lab
                ...

    Retourne le dossier contenant les TextGrids produits.
    """
    input_dir  = Path(input_dir).resolve()
    output_dir = Path(output_dir).resolve()

    audio_dir      = input_dir / "audio"
    transcript_dir = input_dir / "transcriptions"

    # Verifications
    if not audio_dir.exists():
        raise FileNotFoundError(f"Dossier audio introuvable : {audio_dir}")
    if not transcript_dir.exists():
        raise FileNotFoundError(f"Dossier transcriptions introuvable : {transcript_dir}")

    wav_files = sorted(audio_dir.glob("*.wav"))
    if not wav_files:
        raise FileNotFoundError(f"Aucun fichier .wav trouve dans {audio_dir}")

    # Construire le corpus temporaire
    corpus_dir = output_dir / "corpus" / "speaker1"
    corpus_dir.mkdir(parents=True, exist_ok=True)

    paired, missing = [], []
    for wav in wav_files:
        txt = transcript_dir / (wav.stem + ".txt")
        if not txt.exists():
            missing.append(wav.stem)
            continue
        # Copier le .wav
        shutil.copy2(wav, corpus_dir / wav.name)
        # Copier le .txt en .lab (format attendu par MFA)
        shutil.copy2(txt, corpus_dir / (wav.stem + ".lab"))
        paired.append(wav.stem)

    if missing:
        print(f"[MFA] AVERTISSEMENT : {len(missing)} audio(s) sans transcription ignore(s) : {missing}")
    if not paired:
        raise RuntimeError("Aucune paire audio/transcription valide trouvee.")

    print(f"[MFA] Corpus prepare dans : {corpus_dir}")
    print(f"[MFA] {len(paired)} paire(s) audio/transcription : {paired}")

    # Lancer MFA
    tg_output_dir = output_dir / "textgrids"
    tg_output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "mfa", "align",
        str(corpus_dir.parent),   # racine corpus (contient speaker1/)
        dictionary,
        acoustic_model,
        str(tg_output_dir),
        "--clean",
        "--quiet",
    ]

    print(f"[MFA] Commande : {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print("[MFA] STDERR :", result.stderr)
        raise RuntimeError(
            f"MFA a echoue (code {result.returncode}).\n"
            "Verifiez que MFA est installe et que les modeles francais sont telecharges :\n"
            "  mfa model download acoustic french_mfa\n"
            "  mfa model download dictionary french_mfa"
        )

    tg_files = list(tg_output_dir.rglob("*.TextGrid"))
    if not tg_files:
        raise FileNotFoundError(f"Aucun TextGrid produit dans {tg_output_dir}")

    print(f"[MFA] {len(tg_files)} TextGrid(s) genere(s) dans : {tg_output_dir}")
    return tg_output_dir


# -------------------------------------------------
# 2.  LECTURE DU TEXTGRID
# -------------------------------------------------

def list_phonemes(textgrid_path: str) -> list[dict]:
    """
    Lit le TextGrid et retourne tous les intervalles de la tier phones.
    Chaque element : {"phoneme": str, "start": float, "end": float}
    """
    tg = tgio.openTextgrid(textgrid_path, includeEmptyIntervals=False)

    phone_tier_names = [n for n in tg.tierNames if "phone" in n.lower()]
    if not phone_tier_names:
        raise ValueError(
            f"Aucune tier 'phones' trouvee. Tiers disponibles : {tg.tierNames}"
        )

    tier = tg.getTier(phone_tier_names[0])
    return [
        {"phoneme": e.label, "start": e.start, "end": e.end}
        for e in tier.entries
    ]


def print_phonemes(phonemes: list[dict]) -> None:
    """Affiche la liste des phonemes avec leurs timestamps."""
    print(f"\n{'Index':<6} {'Phoneme':<12} {'Debut (s)':<12} {'Fin (s)':<10} {'Duree (ms)'}")
    print("-" * 54)
    for i, p in enumerate(phonemes):
        dur_ms = (p["end"] - p["start"]) * 1000
        marker = "  <- (ch)" if p["phoneme"] == CH_PHONEME else ""
        print(f"{i:<6} {p['phoneme']:<12} {p['start']:<12.4f} {p['end']:<10.4f} {dur_ms:<10.1f}{marker}")
    print()


# -------------------------------------------------
# 3.  EXTRACTION DU SEGMENT AUDIO
# -------------------------------------------------

def extract_phoneme_audio(
    audio_path: str,
    start_sec: float,
    end_sec: float,
    output_wav: str,
    padding_ms: float = 0.0,
) -> None:
    """
    Extrait [start_sec - padding, end_sec + padding] de l'audio
    et sauvegarde en WAV (meme format/sample rate que la source).
    """
    audio, sr = sf.read(audio_path)

    pad          = int(padding_ms / 1000.0 * sr)
    start_sample = max(0, int(start_sec * sr) - pad)
    end_sample   = min(len(audio), int(end_sec * sr) + pad)

    segment = audio[start_sample:end_sample]
    sf.write(output_wav, segment, sr)

    duration_ms = (end_sample - start_sample) / sr * 1000
    print(f"[Export] {output_wav}")
    print(f"         {start_sec:.4f}s -> {end_sec:.4f}s  |  duree : {duration_ms:.1f} ms")


# -------------------------------------------------
# 4.  PIPELINE PRINCIPAL
# -------------------------------------------------

def mfa_pipeline(
    audio_path: str,
    transcript_path: str,
    target_phoneme: str = CH_PHONEME,
    occurrence: int = 1,
    extract_all: bool = False,
    output_dir: str = "mfa_output",
    padding_ms: float = 0.0,
    acoustic_model: str = DEFAULT_ACOUSTIC_MODEL,
    dictionary: str = DEFAULT_DICTIONARY,
    skip_alignment: bool = False,
    textgrid_path: str | None = None,
) -> list[str]:
    """
    Pipeline complet MFA -> extraction phoneme(s) pour un seul fichier.

    Parametres
    ----------
    audio_path      : chemin vers le .wav source
    transcript_path : chemin vers le .txt de transcription
    target_phoneme  : phoneme IPA cible (defaut : ʃ = "ch")
    occurrence      : quelle occurrence extraire (ignore si extract_all=True)
    extract_all     : True pour extraire TOUTES les occurrences
    output_dir      : dossier de sortie
    padding_ms      : marge (ms) ajoutee de chaque cote du segment
    skip_alignment  : True pour utiliser un TextGrid existant
    textgrid_path   : chemin du TextGrid existant (si skip_alignment=True)

    Retourne la liste des chemins WAV extraits.
    """
    audio_path = Path(audio_path).resolve()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Etape 1 : Alignement
    if skip_alignment:
        if not textgrid_path:
            raise ValueError("skip_alignment=True mais textgrid_path non fourni.")
        tg_path = Path(textgrid_path)
        print(f"[Pipeline] TextGrid existant : {tg_path}")
    else:
        # run_mfa_alignment attend input_dir/audio/ et input_dir/transcriptions/
        # On construit un input_dir temporaire a partir des deux fichiers fournis
        transcript_path = Path(transcript_path).resolve()
        tmp_input = output_dir / "tmp_input"
        tmp_audio = tmp_input / "audio"
        tmp_trans = tmp_input / "transcriptions"
        tmp_audio.mkdir(parents=True, exist_ok=True)
        tmp_trans.mkdir(parents=True, exist_ok=True)

        shutil.copy2(audio_path, tmp_audio / audio_path.name)
        shutil.copy2(transcript_path, tmp_trans / (audio_path.stem + ".txt"))

        tg_dir  = run_mfa_alignment(str(tmp_input), str(output_dir), acoustic_model, dictionary)
        matches_tg = list(tg_dir.rglob(audio_path.stem + ".TextGrid"))
        if not matches_tg:
            raise FileNotFoundError(f"TextGrid introuvable pour {audio_path.stem} dans {tg_dir}")
        tg_path = matches_tg[0]

    # Etape 2 : Lire les phonemes
    phonemes = list_phonemes(str(tg_path))
    print_phonemes(phonemes)

    # Etape 3 : Trouver les occurrences du phoneme cible
    matches = [p for p in phonemes if p["phoneme"] == target_phoneme]

    if not matches:
        available = sorted({p["phoneme"] for p in phonemes})
        raise ValueError(
            f"Phoneme '{target_phoneme}' introuvable dans le TextGrid.\n"
            f"Phonemes presents : {available}\n"
            "Astuce : verifiez que votre transcription contient bien des mots avec 'ch'."
        )

    print(f"[Pipeline] {len(matches)} occurrence(s) de '{target_phoneme}' trouvee(s).")

    # Selection des occurrences
    if extract_all:
        to_extract = list(enumerate(matches, start=1))
    else:
        if occurrence > len(matches):
            raise ValueError(
                f"Occurrence {occurrence} demandee, mais seulement "
                f"{len(matches)} occurrence(s) disponible(s)."
            )
        to_extract = [(occurrence, matches[occurrence - 1])]

    # Etape 4 : Exporter les segments
    stem = audio_path.stem
    output_wavs = []

    for occ_num, p in to_extract:
        out_name   = f"{stem}_ch_occ{occ_num}.wav"
        output_wav = str(output_dir / out_name)
        extract_phoneme_audio(str(audio_path), p["start"], p["end"], output_wav, padding_ms)
        output_wavs.append(output_wav)

    return output_wavs


# -------------------------------------------------
# 5.  CLI
# -------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="MFA Phoneme Extractor (Francais) — extrait le son 'ch' (ʃ)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  # Extraire la 1re occurrence du "ch"
  python mfa_phoneme_extractor.py audio.wav transcription.txt

  # Extraire TOUTES les occurrences avec 15ms de marge
  python mfa_phoneme_extractor.py audio.wav transcription.txt --all --padding 15

  # Extraire la 2e occurrence
  python mfa_phoneme_extractor.py audio.wav transcription.txt --occurrence 2

  # Utiliser un TextGrid deja existant (sans re-aligner)
  python mfa_phoneme_extractor.py audio.wav transcription.txt \\
      --skip-alignment --textgrid existing.TextGrid

  # Phoneme personnalise (ex: extraction du [s])
  python mfa_phoneme_extractor.py audio.wav transcription.txt --phoneme s
        """,
    )
    parser.add_argument("audio",      help="Fichier audio source (.wav)")
    parser.add_argument("transcript", help="Fichier de transcription (.txt)")
    parser.add_argument("--phoneme",  default=CH_PHONEME,
                        help="Phoneme IPA cible (defaut : ʃ = 'ch')")
    parser.add_argument("--occurrence", type=int, default=1,
                        help="Quelle occurrence extraire (defaut : 1)")
    parser.add_argument("--all", action="store_true",
                        help="Extraire TOUTES les occurrences du phoneme")
    parser.add_argument("--output-dir", default="mfa_output",
                        help="Dossier de sortie (defaut : mfa_output/)")
    parser.add_argument("--padding", type=float, default=0.0,
                        help="Marge en millisecondes de chaque cote (defaut : 0)")
    parser.add_argument("--acoustic-model", default=DEFAULT_ACOUSTIC_MODEL,
                        help=f"Modele acoustique MFA (defaut : {DEFAULT_ACOUSTIC_MODEL})")
    parser.add_argument("--dictionary", default=DEFAULT_DICTIONARY,
                        help=f"Dictionnaire MFA (defaut : {DEFAULT_DICTIONARY})")
    parser.add_argument("--skip-alignment", action="store_true",
                        help="Sauter MFA et utiliser un TextGrid existant")
    parser.add_argument("--textgrid",
                        help="Chemin d'un TextGrid existant (avec --skip-alignment)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    output_wavs = mfa_pipeline(
        audio_path=args.audio,
        transcript_path=args.transcript,
        target_phoneme=args.phoneme,
        occurrence=args.occurrence,
        extract_all=args.all,
        output_dir=args.output_dir,
        padding_ms=args.padding,
        acoustic_model=args.acoustic_model,
        dictionary=args.dictionary,
        skip_alignment=args.skip_alignment,
        textgrid_path=args.textgrid,
    )

    print(f"\n[OK] {len(output_wavs)} fichier(s) extrait(s) :")
    for wav in output_wavs:
        print(f"   -> {wav}")
