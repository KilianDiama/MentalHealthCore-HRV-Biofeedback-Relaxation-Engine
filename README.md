⚡ Engineered by Kiliandiama | The Diama Protocol [10/10] | All rights reserved.

🧠 MentalHealthCore – HRV Biofeedback & Relaxation Engine

A scientifically inspired Python framework for Heart Rate Variability (HRV) analysis, coherence scoring, and audio-guided relaxation using binaural beats and breathing entrainment.

It is designed for research, prototyping, and biofeedback applications combining signal processing + mental state estimation.

🚀 Features
📊 HRV Analysis (Scientific-grade)
RMSSD computation (time-domain HRV metric)
Welch spectral analysis (LF / HF power)
LF/HF ratio estimation
Bandpass filtering (0.04–0.4 Hz HRV range)
🧹 Signal Cleaning
Automatic artifact detection (ectopic beats, outliers)
Physiological range filtering
Linear interpolation of missing values
🧠 Mental State Scoring
Weighted score (0–100)
60% HRV (RMSSD-based)
40% coherence index
Age-adaptive normalization (optional)
🌊 Coherence Detection
Inspired by heart-brain coherence concepts
Detects dominant frequency stability in 0.04–0.26 Hz band
Peak power ratio estimation
🌬️ Breathing Guidance
Sinusoidal respiratory pacing (default: 6 breaths/min)
Suitable for visual or app-based biofeedback
🎧 Audio Relaxation
Binaural beats generator
Adjustable carrier frequency + beat frequency
Smooth fade-in / fade-out envelope
Compatible with real-time playback (sounddevice)
