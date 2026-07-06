"""
30-Day Breathing Frequency Analysis
Run after simulate.py:
    python analyze.py

Creates:
    ./data/daily_summary.csv
    ./data/analysis_report.txt
    ./data/trend_chart.png
    ./data/status_chart.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_FILE = DATA_DIR / "chestband_data.csv"
DAILY_SUMMARY_FILE = DATA_DIR / "daily_summary.csv"
REPORT_FILE = DATA_DIR / "analysis_report.txt"
TREND_PNG = DATA_DIR / "trend_chart.png"
STATUS_PNG = DATA_DIR / "status_chart.png"

SUSTAINED_HIGH_MINUTES = 10


def find_sustained_high_episodes(df: pd.DataFrame, min_minutes: int = SUSTAINED_HIGH_MINUTES) -> list[dict]:
    """Find continuous periods where breathing status stays High for at least min_minutes."""
    episodes = []
    start_idx = None

    for i, status in enumerate(df["status"]):
        if status == "High" and start_idx is None:
            start_idx = i
        elif status != "High" and start_idx is not None:
            length = i - start_idx
            if length >= min_minutes:
                block = df.iloc[start_idx:i]
                episodes.append(
                    {
                        "start_minute": int(block["simulation_minute"].iloc[0]),
                        "end_minute": int(block["simulation_minute"].iloc[-1]),
                        "start_day": float(block["simulation_day"].iloc[0]),
                        "duration_min": int(length),
                        "max_br": float(block["breathing_rate_bpm"].max()),
                    }
                )
            start_idx = None

    if start_idx is not None:
        block = df.iloc[start_idx:]
        length = len(block)
        if length >= min_minutes:
            episodes.append(
                {
                    "start_minute": int(block["simulation_minute"].iloc[0]),
                    "end_minute": int(block["simulation_minute"].iloc[-1]),
                    "start_day": float(block["simulation_day"].iloc[0]),
                    "duration_min": int(length),
                    "max_br": float(block["breathing_rate_bpm"].max()),
                }
            )

    return episodes


def main() -> None:
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Could not find {DATA_FILE}. Run simulate.py first.")

    df = pd.read_csv(DATA_FILE)
    df["day_number"] = df["simulation_day"].clip(lower=0.0001).apply(lambda x: int((x - 0.0001) // 1) + 1)

    daily = (
        df.groupby("day_number")
        .agg(
            avg_breathing_rate=("breathing_rate_bpm", "mean"),
            min_breathing_rate=("breathing_rate_bpm", "min"),
            max_breathing_rate=("breathing_rate_bpm", "max"),
            high_readings=("status", lambda s: int((s == "High").sum())),
            low_readings=("status", lambda s: int((s == "Low").sum())),
            normal_readings=("status", lambda s: int((s == "Normal").sum())),
        )
        .round(2)
        .reset_index()
    )
    daily.to_csv(DAILY_SUMMARY_FILE, index=False)

    episodes = find_sustained_high_episodes(df)
    worst_day = daily.sort_values(["high_readings", "max_breathing_rate"], ascending=False).iloc[0]

    report = f"""30-Day Breathing Frequency Analysis
-----------------------------------
Total readings: {len(df):,}
Simulation length: {df['simulation_day'].max():.2f} days
Overall average breathing rate: {df['breathing_rate_bpm'].mean():.2f} breaths/min
Overall minimum breathing rate: {df['breathing_rate_bpm'].min():.2f} breaths/min
Overall maximum breathing rate: {df['breathing_rate_bpm'].max():.2f} breaths/min
Normal readings: {(df['status'] == 'Normal').sum():,}
High readings: {(df['status'] == 'High').sum():,}
Low readings: {(df['status'] == 'Low').sum():,}
Sustained high-breathing episodes >= {SUSTAINED_HIGH_MINUTES} min: {len(episodes)}
Most abnormal day: Day {int(worst_day['day_number'])} with {int(worst_day['high_readings'])} high readings

Interpretation:
The dataset represents a 30-day simulated chestband monitoring period. The breathing rate is
estimated from a generated raw chest-expansion signal, then classified using thresholds
Low < 12/min, Normal 12-20/min, High > 20/min. Sustained high episodes are clinically relevant
because they show that a high breathing frequency did not occur as just a single noisy sample.
"""

    REPORT_FILE.write_text(report, encoding="utf-8")
    print(report)
    print(f"Saved daily summary to {DAILY_SUMMARY_FILE}")
    print(f"Saved text report to {REPORT_FILE}")

    plt.figure(figsize=(11, 5))
    plt.plot(df["simulation_day"], df["breathing_rate_bpm"], linewidth=0.8)
    plt.axhline(12, linestyle="--", label="Low threshold: 12/min")
    plt.axhline(20, linestyle="--", label="High threshold: 20/min")
    plt.xlim(0, 30)
    plt.xlabel("Simulation day")
    plt.ylabel("Breathing frequency [breaths/min]")
    plt.title("Breathing Frequency During 30-Day Simulation")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(TREND_PNG, dpi=200)
    plt.close()

    status_counts = df["status"].value_counts().reindex(["Low", "Normal", "High"], fill_value=0)
    plt.figure(figsize=(7, 5))
    status_counts.plot(kind="bar")
    plt.xlabel("Breathing status")
    plt.ylabel("Number of readings")
    plt.title("Breathing Status Distribution")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(STATUS_PNG, dpi=200)
    plt.close()

    print(f"Saved trend chart to {TREND_PNG}")
    print(f"Saved status chart to {STATUS_PNG}")


if __name__ == "__main__":
    main()
