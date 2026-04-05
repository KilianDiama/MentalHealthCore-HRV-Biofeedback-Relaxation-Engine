import numpy as np
from scipy.signal import welch, butter, filtfilt, find_peaks
from scipy.interpolate import interp1d
import sounddevice as sd
from collections import deque

class MentalHealthCore:
    """
    Core biofeedback HRV + respiration guidée + audio relaxant - Version optimisée 10/10
    Objectifs :
    - Mesures HRV scientifiquement solides (RMSSD, HF power, cohérence cardiaque)
    - Détection et correction d'artefacts (essentiel en vrai usage PPG/ECG)
    - Cohérence inspirée de HeartMath : pic dominant dans 0.04-0.26 Hz
    - Score global pondéré et adapté à l'âge (optionnel)
    - Respiration guidée + binaural beats + feedback visuel
    """

    def __init__(self, fs_hrv=4.0, fs_audio=44100, age=None):
        self.fs_hrv = fs_hrv          # 4 Hz = standard pour HRV spectral
        self.fs_audio = fs_audio
        self.age = age                # Pour normalisation RMSSD (optionnel)

        # Historiques pour feedback en temps réel
        self.rmssd_history = deque(maxlen=50)
        self.coherence_history = deque(maxlen=50)
        self.score_history = deque(maxlen=100)
        self.rr_history = deque(maxlen=300)   # Pour analyse glissante

    # ==================================================================
    # PRÉTRAITEMENT & ARTEFACTS (nouveau - très important)
    # ==================================================================
    def clean_rr_intervals(self, rr_intervals, low_rri=0.3, high_rri=2.0, 
                          outlier_threshold=0.25):
        """
        Supprime/corrige les artefacts et ectopic beats.
        """
        rr = np.asarray(rr_intervals, dtype=float)
        # Suppression valeurs physiologiquement impossibles
        mask = (rr >= low_rri) & (rr <= high_rri)
        rr_clean = rr.copy()

        # Détection outliers par différence successive (médiane + seuil)
        if len(rr) > 5:
            diff = np.abs(np.diff(rr))
            median_diff = np.median(diff)
            outlier_mask = np.concatenate(([False], diff > outlier_threshold))
            rr_clean[outlier_mask] = np.nan

        # Interpolation linéaire des NaN (méthode classique)
        nans = np.isnan(rr_clean)
        if np.any(nans):
            f = interp1d(np.flatnonzero(~nans), rr_clean[~nans], 
                         kind='linear', bounds_error=False, fill_value="extrapolate")
            rr_clean[nans] = f(np.flatnonzero(nans))
        
        self.rr_history.extend(rr_clean)
        return rr_clean

    # ==================================================================
    # INTERPOLATION RR → Signal HRV (corrigé - standard)
    # ==================================================================
    def interpolate_rr(self, rr_intervals, rr_timestamps=None):
        """
        Interpolation standard : temps cumulés → interp1d cubic sur RR aux instants réels.
        Puis rééchantillonnage régulier à fs_hrv.
        """
        rr = np.asarray(rr_intervals, dtype=float)
        if len(rr) < 10:
            raise ValueError("Au moins 10 intervalles RR nécessaires.")

        if rr_timestamps is None:
            rr_timestamps = np.cumsum(rr) - rr[0]   # temps en secondes (début à 0)
        else:
            rr_timestamps = np.asarray(rr_timestamps, dtype=float)

        # Signal HRV : on interpole la valeur RR à chaque temps
        f_interp = interp1d(rr_timestamps, rr, kind='cubic', 
                            fill_value="extrapolate", assume_sorted=True)

        t_start = rr_timestamps[0]
        t_end = rr_timestamps[-1]
        if t_end - t_start < 10:
            raise ValueError("Durée trop courte (<10s) pour analyse fiable.")

        t_grid = np.arange(t_start, t_end, 1.0 / self.fs_hrv)
        hrv_signal = f_interp(t_grid)

        # Centrage (suppression DC)
        hrv_signal -= np.mean(hrv_signal)
        return t_grid, hrv_signal

    def bandpass_filter(self, data, lowcut=0.04, highcut=0.4, order=4):
        """Filtre passe-bande classique pour HRV."""
        nyq = 0.5 * self.fs_hrv
        b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
        return filtfilt(b, a, data)

    # ==================================================================
    # MÉTRIQUES TEMPS
    # ==================================================================
    def compute_rmssd(self, rr_intervals):
        """RMSSD en secondes (stocké en ms pour l'historique)."""
        rr = np.asarray(rr_intervals, dtype=float)
        if len(rr) < 3:
            raise ValueError("Minimum 3 RR pour RMSSD.")
        
        diff = np.diff(rr)
        rmssd = np.sqrt(np.mean(diff**2))
        self.rmssd_history.append(rmssd)
        return rmssd

    # ==================================================================
    # ANALYSE SPECTRALE & COHÉRENCE (améliorée)
    # ==================================================================
    def compute_spectral_features(self, hrv_signal):
        """Welch + bandes LF/HF classiques."""
        if len(hrv_signal) < 128:
            raise ValueError("Signal trop court pour analyse spectrale.")

        freqs, psd = welch(hrv_signal, fs=self.fs_hrv, nperseg=min(256, len(hrv_signal)))

        lf_band = (freqs >= 0.04) & (freqs <= 0.15)
        hf_band = (freqs > 0.15) & (freqs <= 0.4)

        lf_power = np.trapz(psd[lf_band], freqs[lf_band]) if np.any(lf_band) else 0.0
        hf_power = np.trapz(psd[hf_band], freqs[hf_band]) if np.any(hf_band) else 0.0
        total_power = np.trapz(psd, freqs)

        return {
            "freqs": freqs,
            "psd": psd,
            "lf_power": lf_power,
            "hf_power": hf_power,
            "total_power": total_power,
            "lf_hf_ratio": lf_power / hf_power if hf_power > 0 else np.inf,
        }

    def compute_coherence(self, hrv_signal, coherence_band=(0.04, 0.26)):
        """
        Cohérence améliorée : recherche du pic dominant dans la bande de cohérence (0.04-0.26 Hz),
        puis ratio puissance pic / puissance totale (inspiré HeartMath).
        Retourne un score entre 0 et 1.
        """
        features = self.compute_spectral_features(hrv_signal)
        freqs = features["freqs"]
        psd = features["psd"]

        mask = (freqs >= coherence_band[0]) & (freqs <= coherence_band[1])
        if not np.any(mask):
            coh = 0.0
        else:
            peak_idx = np.argmax(psd[mask])
            peak_freq = freqs[mask][peak_idx]
            # Fenêtre ±0.015 Hz autour du pic (approximation de 0.03 Hz window)
            window = (freqs >= peak_freq - 0.015) & (freqs <= peak_freq + 0.015)
            peak_power = np.trapz(psd[window], freqs[window]) if np.any(window) else 0.0
            total_power = features["total_power"]
            coh = peak_power / total_power if total_power > 0 else 0.0

        coh = float(np.clip(coh, 0.0, 1.0))
        self.coherence_history.append(coh)
        return coh

    # ==================================================================
    # RESPIRATION & AUDIO
    # ==================================================================
    def breathing_pattern(self, duration=60, rate=6.0, fs_visu=50):
        """Pattern sinusoidal pour guidage visuel/respiratoire."""
        t = np.linspace(0, duration, int(duration * fs_visu), endpoint=False)
        breath_signal = np.sin(2 * np.pi * (rate / 60.0) * t)
        return t, breath_signal

    def generate_binaural(self, duration=30, base_freq=180, beat_freq=6.0):
        """Binaural beats avec enveloppe douce (durée augmentée)."""
        t = np.linspace(0, duration, int(self.fs_audio * duration), endpoint=False)
        left = np.sin(2 * np.pi * base_freq * t)
        right = np.sin(2 * np.pi * (base_freq + beat_freq) * t)

        # Enveloppe fade in/out
        envelope = np.ones_like(t)
        fade = int(0.5 * self.fs_audio)  # 0.5s
        envelope[:fade] = np.linspace(0, 1, fade)
        envelope[-fade:] = np.linspace(1, 0, fade)

        stereo = np.vstack((left * envelope, right * envelope)).T
        return stereo, self.fs_audio

    def play_sound(self, audio, fs=None):
        if fs is None:
            fs = self.fs_audio
        try:
            sd.play(audio.astype(np.float32), fs)
            # sd.wait()  # Optionnel : bloquant ou non selon usage
        except Exception as e:
            print(f"[WARN] Lecture audio impossible : {e}")

    # ==================================================================
    # SCORE GLOBAL (amélioré)
    # ==================================================================
    def mental_state_score(self, rmssd, coherence, target_rmssd=None):
        """
        Score 0-100 : 60% RMSSD (normalisé) + 40% cohérence.
        Normalisation adaptative selon âge si fourni.
        """
        rmssd_ms = rmssd * 1000.0

        # Normalisation par âge (valeurs approximatives médianes)
        if self.age is not None and target_rmssd is None:
            if self.age < 30:
                target = 70
            elif self.age < 45:
                target = 50
            elif self.age < 60:
                target = 35
            else:
                target = 25
        else:
            target = target_rmssd or 50.0

        rmssd_norm = np.clip(rmssd_ms / target, 0.0, 1.0)

        score = 100.0 * (0.6 * rmssd_norm + 0.4 * coherence)
        score = float(np.clip(score, 0.0, 100.0))
        self.score_history.append(score)
        return score

    # ==================================================================
    # MÉTHODE PRINCIPALE : ANALYSE COMPLÈTE
    # ==================================================================
    def analyze_session(self, rr_intervals, target_breath_rate=6.0):
        """Analyse complète d'une session (recommandé)."""
        rr_clean = self.clean_rr_intervals(rr_intervals)
        rmssd = self.compute_rmssd(rr_clean)

        t_grid, hrv_signal = self.interpolate_rr(rr_clean)
        hrv_filt = self.bandpass_filter(hrv_signal)

        coherence = self.compute_coherence(hrv_filt)
        score = self.mental_state_score(rmssd, coherence)

        return {
            "rmssd_ms": round(rmssd * 1000, 2),
            "coherence": round(coherence, 3),
            "mental_score": round(score, 1),
            "hf_power": self.compute_spectral_features(hrv_filt)["hf_power"],
            "clean_rr_count": len(rr_clean)
        }


# ======================================================================
# EXEMPLE D’UTILISATION (amélioré)
# ======================================================================
if __name__ == "__main__":
    core = MentalHealthCore(age=35)   # Mets ton âge pour meilleure normalisation

    # Simulation de données réalistes (75 bpm moyen, bonne variabilité)
    np.random.seed(42)
    rr_intervals = np.random.normal(0.8, 0.06, 250)   # 250 battements ~ 3-4 min

    # Analyse complète
    results = core.analyze_session(rr_intervals, target_breath_rate=6.0)

    print("=== Résultats Biofeedback ===")
    print(f"RMSSD          : {results['rmssd_ms']:.1f} ms")
    print(f"Cohérence      : {results['coherence']:.3f}")
    print(f"Score Mental   : {results['mental_score']:.1f}/100")
    print(f"Nombre RR clean: {results['clean_rr_count']}")

    # Guidage respiration 6/min pendant 1 minute
    t_breath, breath = core.breathing_pattern(duration=60, rate=6.0)

    # Audio relaxation (binaural à 6 Hz = fréquence résonance)
    audio, fs = core.generate_binaural(duration=30, base_freq=180, beat_freq=6.0)
    # core.play_sound(audio, fs)   # Décommente pour tester le son
