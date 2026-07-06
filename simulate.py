"""
Chestband ECG + Breathing Data Simulator
Project: Chestband with ECG and App Readout - 30-Day Breathing Frequency Analysis

Run this first:
    python simulate.py

Creates:
    ./data/chestband_data.csv

What this simulates:
- Heart rate as a realistic minute-by-minute vital sign (daily rhythm + activity + noise).
- A raw chest-expansion (respiration) signal, the way a real chestband sensor would output it.
- Breathing rate is ESTIMATED from that raw signal (peak detection), not just stored as a
  made-up number. This mirrors how the real device / app pipeline is supposed to work.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# Everything lives next to this script, in a "data" folder. Every other script in this
# project points at the same folder, so nothing has to guess a parent/sibling layout.
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_FILE = DATA_DIR / "chestband_data.csv"

SEED = 42
TOTAL_DAYS = 30
MINUTES_PER_DAY = 24 * 60
TOTAL_MINUTES = TOTAL_DAYS * MINUTES_PER_DAY

BASE_HEART_RATE = 75.0
BASE_BREATHING_RATE = 16.0
RESP_FS = 5          # raw chestband samples per second
RESP_SECONDS = 60    # one minute of raw signal generated for each row

rng = np.random.default_rng(SEED)


def classify_breathing(rate: float) -> str:
    """Classify adult resting breathing frequency using simple project thresholds.

    Step 4 of the selected signal processing chain: rate estimate -> CLASSIFICATION
    (Low/Normal/High). Step 5, the live readout, happens in monitor.py.
    """

    if rate < 12:
        return "Low"
    if rate <= 20:
        return "Normal"
    return "High"


def simulate_heart_rate(minute: int) -> float:
    """Realistic heart rate: daily rhythm + activity bumps + noise."""
    day_fraction = (minute % MINUTES_PER_DAY) / MINUTES_PER_DAY
    circadian = 4 * np.sin(2 * np.pi * (day_fraction - 0.25))
    noise = rng.normal(0, 2)

    hour = (minute % MINUTES_PER_DAY) / 60
    activity_boost = 0.0
    if 8 <= hour <= 10 or 17 <= hour <= 19:
        activity_boost = rng.uniform(4, 12)

    return float(np.clip(BASE_HEART_RATE + circadian + activity_boost + noise, 55, 115))


def true_breathing_rate(minute: int) -> float:
    """Hidden 'true' breathing rate used to generate the raw chest signal."""
    day_fraction = (minute % MINUTES_PER_DAY) / MINUTES_PER_DAY
    daily_rhythm = 1.8 * np.sin(2 * np.pi * (day_fraction - 0.15))
    noise = rng.normal(0, 0.7)
    rate = BASE_BREATHING_RATE + daily_rhythm + noise

    # Sustained high-breathing episodes so the analysis has something clinically meaningful.
    if minute % 7000 in range(100, 250):
        rate += rng.uniform(5, 8)

    # Rare low-breathing dips at night.
    hour = (minute % MINUTES_PER_DAY) / 60
    if 1 <= hour <= 4 and rng.random() < 0.015:
        rate -= rng.uniform(3, 5)

    return float(np.clip(rate, 8, 30))


def simulate_chest_expansion_signal(rate_bpm: float) -> np.ndarray:
    """Simulate the raw chestband respiration signal for one minute.
    
    This is Step 1 of the selected signal processing chain (see the technical report):
    hidden true rate -> RAW SIGNAL -> peak detection -> rate estimate -> classification.
    """
    
    t = np.arange(RESP_SECONDS * RESP_FS) / RESP_FS
    freq_hz = rate_bpm / 60.0

    signal = np.sin(2 * np.pi * freq_hz * t)                       # main breathing wave
    signal += 0.15 * np.sin(2 * np.pi * 2 * freq_hz * t + 0.4)      # small harmonic
    signal += 0.08 * np.sin(2 * np.pi * 0.03 * t)                   # slow baseline drift
    signal += rng.normal(0, 0.08, size=len(t))                      # sensor noise
    return signal


def estimate_breathing_rate_from_chest_signal(signal: np.ndarray, fs: int = RESP_FS) -> float:
    """Estimate breathing rate from raw chest-expansion peaks (simple peak detector).

    This is Step 2 of the selected signal processing chain: raw signal -> PEAK DETECTION
    -> rate estimate. A candidate sample only counts as a breath if it clears two
    conditions at once (both explained in the report's Theory section):
      1. It is tall enough (is_large_enough) - rejects small sensor-noise wiggles.
      2. It is far enough in time from the last accepted peak (far_enough) - rejects
         double-counting the same breath and rejects fast artifacts like heartbeat
         "cardifacts" bleeding into the respiration signal.
    """

    centered = signal - np.mean(signal)
    peaks = []

    min_distance_samples = int(fs * 1.5)  # caps detection at ~40 breaths/min
    last_peak = -min_distance_samples

    for i in range(1, len(centered) - 1):
        is_peak = centered[i] > centered[i - 1] and centered[i] > centered[i + 1]
        is_large_enough = centered[i] > 0.35
        far_enough = (i - last_peak) >= min_distance_samples
        if is_peak and is_large_enough and far_enough:
            peaks.append(i)
            last_peak = i

    if len(peaks) >= 2:
        rr_seconds = np.diff(peaks) / fs
        return float(60.0 / np.mean(rr_seconds))  # Step 3: peaks -> rate estimate (breaths/min)

    # Fallback: count upward zero-crossings.
    crossings = np.where((centered[:-1] < 0) & (centered[1:] >= 0))[0]
    breaths = max(len(crossings), 1)
    return float(breaths * 60 / RESP_SECONDS)


def create_dataset(total_minutes: int = TOTAL_MINUTES, verbose: bool = True) -> pd.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    start_time = datetime.now().replace(microsecond=0)
    records = []

    for minute in range(total_minutes):
        if verbose and minute % 5000 == 0:
            print(f"  simulating minute {minute:,} / {total_minutes:,} "
                  f"(day {minute / MINUTES_PER_DAY:.1f})")

        hr = simulate_heart_rate(minute)
        true_br = true_breathing_rate(minute)
        chest_signal = simulate_chest_expansion_signal(true_br)
        estimated_br = estimate_breathing_rate_from_chest_signal(chest_signal)
        status = classify_breathing(estimated_br)

        records.append(
            {
                "timestamp": (start_time + timedelta(minutes=minute)).strftime("%Y-%m-%d %H:%M:%S"),
                "simulation_minute": minute,
                "simulation_day": round((minute + 1) / MINUTES_PER_DAY, 4),
                "heart_rate_bpm": round(hr, 2),
                "true_breathing_rate_bpm": round(true_br, 2),
                "breathing_rate_bpm": round(estimated_br, 2),
                "chest_expansion_mean": round(float(np.mean(chest_signal)), 5),
                "chest_expansion_std": round(float(np.std(chest_signal)), 5),
                "status": status,
            }
        )

    return pd.DataFrame(records)


def main() -> None:
    print(f"Simulating {TOTAL_DAYS} days ({TOTAL_MINUTES:,} minutes) of chestband data...")
    df = create_dataset()
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"\nSaved {len(df):,} rows to {OUTPUT_FILE}")
    print(f"Simulation length: {df['simulation_day'].max():.2f} days")
    print("\nBreathing summary:")
    print(df["breathing_rate_bpm"].describe().round(2))
    print("\nStatus counts:")
    print(df["status"].value_counts())


if __name__ == "__main__":
    main()
