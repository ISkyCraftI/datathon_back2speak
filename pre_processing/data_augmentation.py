"""
Audio Data Augmentation — Spectral Centroid & High-Frequency Energy
====================================================================
Objectif : augmenter un dataset audio pour la reconnaissance vocale
en modifiant subtilement les caractéristiques spectrales du signal,
tout en conservant l'intelligibilité humaine.

Dépendances :
    pip install librosa numpy scipy soundfile
"""

import numpy as np
import librosa
import librosa.effects
import soundfile as sf
from scipy.signal import butter, sosfilt

def shift_spectral_centroid(
    y: np.ndarray,
    sr: int,
    shift_factor: float = 1.10,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> np.ndarray:
    """
    Déplace le centroïde spectral vers les hautes ou basses fréquences
    en redistribuant l'énergie dans le spectre via un pondération fréquentielle.

    Principe :
        On calcule le STFT, on applique un masque de pondération linéaire
        sur les fréquences (amplifie / atténue selon shift_factor), puis
        on reconstruit le signal par ISTFT (Griffin-Lim ou phase originale).

    Args:
        y            : signal audio (float32 normalisé [-1, 1])
        sr           : taux d'échantillonnage (Hz)
        shift_factor : > 1.0 → centroïde vers les aigus
                       < 1.0 → centroïde vers les graves
                       Plage recommandée : [0.85, 1.20]
        n_fft        : taille de la FFT
        hop_length   : pas entre trames

    Returns:
        y_aug : signal augmenté (même durée, même sr)
    """
    assert 0.70 <= shift_factor <= 1.40, (
        "shift_factor hors plage sûre [0.70, 1.40] — risque d'inintelligibilité"
    )

    # --- STFT complexe (conserve la phase) ---
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    magnitude, phase = np.abs(D), np.angle(D)

    n_bins = magnitude.shape[0]  # nombre de bins fréquentiels

    # --- Masque de pondération fréquentielle ---
    # Rampe linéaire : bins graves → poids faibles si shift > 1 (et inversement)
    # On normalise pour conserver l'énergie globale (RMS stable)
    ramp = np.linspace(1.0 / shift_factor, shift_factor, n_bins)  # (n_bins,)
    ramp = ramp[:, np.newaxis]                                      # broadcast sur les trames

    magnitude_shifted = magnitude * ramp

    # --- Normalisation RMS pour éviter la distorsion de volume ---
    rms_orig = np.sqrt(np.mean(magnitude ** 2))
    rms_new  = np.sqrt(np.mean(magnitude_shifted ** 2))
    if rms_new > 1e-8:
        magnitude_shifted *= rms_orig / rms_new

    # --- Reconstruction avec la phase originale ---
    D_aug = magnitude_shifted * np.exp(1j * phase)
    y_aug = librosa.istft(D_aug, hop_length=hop_length, length=len(y))

    return y_aug.astype(np.float32)


def modify_high_frequency_energy(
    y: np.ndarray,
    sr: int,
    gain_db: float = 4.0,
    cutoff_hz: float = 4000.0,
    order: int = 4,
) -> np.ndarray:
    """
    Amplifie ou atténue l'énergie au-dessus de `cutoff_hz` via un filtre
    passe-haut de Butterworth appliqué en parallèle (shelving effect).

    Principe :
        y_aug = y_low + gain_linear * y_high
        où y_low  = y - y_high  (composantes sous cutoff_hz)
              y_high = filtre passe-haut(y)  (composantes au-dessus)

    Args:
        y          : signal audio (float32 normalisé [-1, 1])
        sr         : taux d'échantillonnage (Hz)
        gain_db    : gain appliqué aux HF en dB
                     > 0 → booste les sibilantes/fricatives (s, f, ch…)
                     < 0 → adoucit les HF
                     Plage recommandée : [-6, +6] dB
        cutoff_hz  : fréquence de coupure (Hz). 4000 Hz est un bon seuil
                     pour la parole (consonnes fricatives)
        order      : ordre du filtre Butterworth (4 = pente douce)

    Returns:
        y_aug : signal avec énergie HF modifiée
    """
    assert -12 <= gain_db <= 12, (
        "gain_db hors plage sûre [-12, +12] dB — risque de distorsion"
    )
    assert cutoff_hz < sr / 2, (
        f"cutoff_hz ({cutoff_hz}) doit être < Nyquist ({sr/2})"
    )

    gain_linear = 10 ** (gain_db / 20.0)

    # --- Filtre passe-haut Butterworth ---
    nyq = sr / 2.0
    sos = butter(order, cutoff_hz / nyq, btype="high", output="sos")
    y_high = sosfilt(sos, y).astype(np.float32)

    # --- Recomposition : basses fréquences intactes + HF pondérées ---
    y_low  = (y - y_high).astype(np.float32)
    y_aug  = y_low + gain_linear * y_high

    # --- Clipping doux pour éviter la saturation ---
    y_aug = np.clip(y_aug, -1.0, 1.0)

    return y_aug

def augment_audio(
    y: np.ndarray,
    sr: int,
    # Centroïde spectral
    centroid_shift: float = 1.08,
    # Énergie haute fréquence
    hf_gain_db: float = 3.0,
    hf_cutoff_hz: float = 4000.0,
    # Ordre d'application
    apply_centroid_first: bool = True,
) -> np.ndarray:
    """
    Pipeline complet d'augmentation spectrale pour la reconnaissance vocale.

    Combine dans un ordre configurable :
      1. Déplacement du centroïde spectral
      2. Modification de l'énergie haute fréquence

    Args:
        y                   : signal audio brut
        sr                  : taux d'échantillonnage
        centroid_shift      : facteur de déplacement du centroïde [0.85–1.20]
        hf_gain_db          : gain HF en dB [-6, +6]
        hf_cutoff_hz        : fréquence de coupure pour les HF
        apply_centroid_first: ordre des transformations

    Returns:
        y_aug : signal augmenté, même durée, même sr
    """
    ops = [
        lambda sig: shift_spectral_centroid(sig, sr, shift_factor=centroid_shift),
        lambda sig: modify_high_frequency_energy(sig, sr, gain_db=hf_gain_db,
                                                 cutoff_hz=hf_cutoff_hz),
    ]
    if not apply_centroid_first:
        ops = ops[::-1]

    y_aug = y.copy().astype(np.float32)
    for op in ops:
        y_aug = op(y_aug)

    return y_aug

def generate_augmented_variants(
    y: np.ndarray,
    sr: int,
    n_variants: int = 5,
    seed: int = 42,
) -> list[dict]:
    """
    Génère N variantes augmentées avec des paramètres aléatoires
    dans les plages sûres (intelligibilité garantie).

    Args:
        y          : signal audio source
        sr         : taux d'échantillonnage
        n_variants : nombre de variantes à produire
        seed       : graine aléatoire pour la reproductibilité

    Returns:
        Liste de dicts : {"audio": np.ndarray, "params": dict}
    """
    rng = np.random.default_rng(seed)
    variants = []

    for i in range(n_variants):
        params = {
            "centroid_shift": float(rng.uniform(0.88, 1.15)),
            "hf_gain_db":     float(rng.uniform(-4.0, 5.0)),
            "hf_cutoff_hz":   float(rng.choice([3500, 4000, 4500, 5000])),
            "apply_centroid_first": bool(rng.integers(0, 2)),
        }
        y_aug = augment_audio(y, sr, **params)
        variants.append({"audio": y_aug, "params": params})
        print(f"  Variante {i+1:02d} | centroid_shift={params['centroid_shift']:.3f} "
              f"| hf_gain={params['hf_gain_db']:+.1f} dB "
              f"| cutoff={params['hf_cutoff_hz']:.0f} Hz")

    return variants


# ─────────────────────────────────────────────
#  5. EXEMPLE D'UTILISATION
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import os

    # --- Chargement d'un fichier audio ---
    AUDIO_PATH = "chat.wav"   # ← remplacer par votre fichier

    if not os.path.exists(AUDIO_PATH):
        # Génère un signal de test si aucun fichier n'est fourni
        print("Aucun fichier audio trouvé — génération d'un signal de test...")
        sr_test = 22050
        duration = 2.0  # secondes
        t = np.linspace(0, duration, int(sr_test * duration), endpoint=False)
        # Signal vocal synthétique : fondamentale 150 Hz + harmoniques
        y_test = (
            0.5  * np.sin(2 * np.pi * 150  * t) +
            0.3  * np.sin(2 * np.pi * 300  * t) +
            0.15 * np.sin(2 * np.pi * 600  * t) +
            0.05 * np.sin(2 * np.pi * 3000 * t)
        ).astype(np.float32)
        y_orig, sr = y_test, sr_test
    else:
        y_orig, sr = librosa.load(AUDIO_PATH, sr=None, mono=True)
        print(f"Fichier chargé : {AUDIO_PATH} | sr={sr} Hz | durée={len(y_orig)/sr:.2f}s")

    # --- Exemple 1 : modification du centroïde seule ---
    y_centroid = shift_spectral_centroid(y_orig, sr, shift_factor=1.10)
    sf.write("out_centroid_shift.wav", y_centroid, sr)
    print("→ out_centroid_shift.wav")

    # --- Exemple 2 : modification des HF seule ---
    y_hf = modify_high_frequency_energy(y_orig, sr, gain_db=4.0, cutoff_hz=4000)
    sf.write("out_hf_boost.wav", y_hf, sr)
    print("→ out_hf_boost.wav")

    # --- Exemple 3 : pipeline combiné ---
    y_combined = augment_audio(
        y_orig, sr,
        centroid_shift=1.08,
        hf_gain_db=3.5,
        hf_cutoff_hz=4000,
    )
    sf.write("out_combined.wav", y_combined, sr)
    print("→ out_combined.wav")

    # --- Exemple 4 : génération de N variantes ---
    print("\nGénération de 5 variantes aléatoires :")
    variants = generate_augmented_variants(y_orig, sr, n_variants=5, seed=0)
    for i, v in enumerate(variants):
        sf.write(f"out_variant_{i+1:02d}.wav", v["audio"], sr)
    print("Variantes sauvegardées.")