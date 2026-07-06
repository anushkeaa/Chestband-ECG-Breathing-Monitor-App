"""
Chestband ECG & Breathing Monitor App
Project: Chestband with ECG and App Readout - 30-Day Breathing Frequency Analysis

Run order:
    1. python simulate.py
    2. python monitor.py

Important project note:
- This is a simulation dashboard, not a certified medical device.
- ECG is synthetically generated for visualization.
- Heart rate is also estimated from the synthetic ECG R-peaks, to show the ECG -> HR concept.
- Breathing rate comes from simulated raw chest-expansion data generated in simulate.py.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "chestband_data.csv"

BG = "#071018"
CARD = "#0d1822"
TEXT = "white"
GREEN = "#00ff3c"
BLUE = "#149cff"
RED = "#ff3b30"
YELLOW = "#ffd60a"
GRAY = "#a9b4c2"

BASE_HEART_RATE = 75.0
SUSTAINED_HIGH_MINUTES = 10


class ECGMonitorApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Chestband ECG & Breathing Monitor")
        self.root.geometry("1250x800")
        self.root.configure(bg=BG)

        # ---- ECG buffer setup ----
        self.fs = 250
        self.buffer_seconds = 5
        self.buffer_size = self.buffer_seconds * self.fs
        self.ecg_x = np.linspace(0, self.buffer_seconds, self.buffer_size)
        self.ecg_phase = self.buffer_size
        self.row_index = 0

        self.ecg_buffer = np.array(
            [self.compute_ecg_sample(BASE_HEART_RATE, i) for i in range(self.buffer_size)],
            dtype=float,
        )

        # ---- Respiration buffer setup ----
        # FIX: the old version rebuilt the whole wave from t=0 every single frame using
        # np.linspace(0, 20, 400) and a fresh sin(2*pi*freq*t). Because it always restarted
        # the phase at t=0, the two ends of the plot kept snapping back to the same spot on
        # every redraw while only the middle bulged - that's the "harmonium" look. The fix
        # is the same trick already used for the ECG: keep one persistent buffer, advance a
        # running phase, and roll the newest sample in. That makes it scroll continuously
        # instead of resetting every frame.
        self.resp_fs = 20                        # respiration samples per second (visual, not physiological)
        self.resp_buffer_seconds = 20
        self.resp_buffer_size = self.resp_fs * self.resp_buffer_seconds
        self.resp_x = np.linspace(0, self.resp_buffer_seconds, self.resp_buffer_size)
        self.resp_phase = 0.0
        self.resp_buffer = np.zeros(self.resp_buffer_size)

        title = tk.Label(root, text="Chestband ECG & Breathing Monitor", font=("Arial", 23, "bold"), fg=TEXT, bg=BG)
        title.pack(pady=10)

        main = tk.Frame(root, bg=BG)
        main.pack(fill="both", expand=True, padx=12, pady=8)

        left = tk.Frame(main, bg=BG)
        left.pack(side="left", fill="both", expand=True, padx=(0, 12))

        right = tk.Frame(main, bg=CARD, width=350)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)

        self.ecg_fig = Figure(figsize=(8, 3), dpi=100, facecolor=CARD)
        self.ecg_ax = self.ecg_fig.add_subplot(111)
        self.ecg_canvas = FigureCanvasTkAgg(self.ecg_fig, master=left)
        self.ecg_canvas.get_tk_widget().pack(fill="both", expand=True, pady=(0, 8))

        self.resp_fig = Figure(figsize=(8, 2.4), dpi=100, facecolor=CARD)
        self.resp_ax = self.resp_fig.add_subplot(111)
        self.resp_canvas = FigureCanvasTkAgg(self.resp_fig, master=left)
        self.resp_canvas.get_tk_widget().pack(fill="both", expand=True, pady=(0, 8))

        self.trend_fig = Figure(figsize=(8, 2.8), dpi=100, facecolor=CARD)
        self.trend_ax = self.trend_fig.add_subplot(111)
        self.trend_canvas = FigureCanvasTkAgg(self.trend_fig, master=left)
        self.trend_canvas.get_tk_widget().pack(fill="both", expand=True)

        tk.Label(right, text="Heart Rate from CSV", font=("Arial", 13, "bold"), fg=GRAY, bg=CARD).pack(pady=(22, 0))
        self.hr_label = tk.Label(right, text="-- bpm", font=("Arial", 30, "bold"), fg=RED, bg=CARD)
        self.hr_label.pack(pady=(0, 5))

        tk.Label(right, text="Estimated HR from ECG Peaks", font=("Arial", 13, "bold"), fg=GRAY, bg=CARD).pack(pady=(4, 0))
        self.hr_est_label = tk.Label(right, text="-- bpm", font=("Arial", 22, "bold"), fg=GREEN, bg=CARD)
        self.hr_est_label.pack(pady=(0, 12))

        tk.Label(right, text="Breathing Rate", font=("Arial", 13, "bold"), fg=GRAY, bg=CARD).pack(pady=(3, 0))
        self.br_label = tk.Label(right, text="-- /min", font=("Arial", 30, "bold"), fg=BLUE, bg=CARD)
        self.br_label.pack(pady=(0, 8))

        tk.Label(right, text="Status", font=("Arial", 13, "bold"), fg=GRAY, bg=CARD).pack(pady=(3, 0))
        self.status = tk.Label(right, text="WAITING", font=("Arial", 24, "bold"), fg=YELLOW, bg=CARD)
        self.status.pack(pady=(0, 16))

        self.alert = tk.Label(right, text="No sustained alert", font=("Arial", 13, "bold"), fg=GREEN, bg=CARD, wraplength=300)
        self.alert.pack(pady=(0, 14))

        self.info = tk.Label(right, text="", font=("Arial", 11), fg=TEXT, bg=CARD, justify="left", anchor="w")
        self.info.pack(fill="x", padx=22)

        self.update_app()

    def compute_ecg_sample(self, heart_rate: float, sample_index: int) -> float:
        """Generate one synthetic ECG sample. R-peaks repeat according to HR."""
        beat_interval = 60.0 / max(heart_rate, 1.0)
        t = (sample_index / self.fs) % beat_interval

        value = 0.0
        value += 1.20 * np.exp(-((t - 0.08) ** 2) / (2 * 0.010**2))   # R peak
        value += -0.25 * np.exp(-((t - 0.10) ** 2) / (2 * 0.008**2))  # S dip
        value += 0.12 * np.exp(-((t - 0.22) ** 2) / (2 * 0.040**2))   # T wave
        value += 0.02 * np.random.normal()
        return float(value)

    def estimate_hr_from_rr(self, ecg_buffer: np.ndarray) -> float | None:
        """Estimate HR from synthetic ECG R-peak distances."""
        peaks = []
        min_distance = int(0.35 * self.fs)
        last_peak = -min_distance

        for i in range(1, len(ecg_buffer) - 1):
            is_peak = ecg_buffer[i] > 0.8 and ecg_buffer[i] > ecg_buffer[i - 1] and ecg_buffer[i] > ecg_buffer[i + 1]
            far_enough = (i - last_peak) >= min_distance
            if is_peak and far_enough:
                peaks.append(i)
                last_peak = i

        if len(peaks) >= 2:
            rr_mean = np.mean(np.diff(peaks)) / self.fs
            return float(60.0 / rr_mean)
        return None

    def draw_ecg(self, heart_rate: float) -> float | None:
        new_value = self.compute_ecg_sample(heart_rate, self.ecg_phase)
        self.ecg_phase += 1
        self.ecg_buffer[:-1] = self.ecg_buffer[1:]
        self.ecg_buffer[-1] = new_value
        estimated_hr = self.estimate_hr_from_rr(self.ecg_buffer)

        self.ecg_ax.clear()
        self.ecg_ax.set_facecolor("black")
        self.ecg_ax.plot(self.ecg_x, self.ecg_buffer, color=GREEN, linewidth=1.5)
        self.ecg_ax.set_title("Live Simulated ECG Waveform with R-Peak HR Estimation", color=TEXT, fontsize=12)
        self.ecg_ax.set_xlim(0, self.buffer_seconds)
        self.ecg_ax.set_ylim(-1.5, 1.5)
        self.ecg_ax.set_xticks(np.arange(0, self.buffer_seconds + 0.1, 0.2), minor=True)
        self.ecg_ax.set_yticks(np.arange(-1.5, 1.6, 0.2), minor=True)
        self.ecg_ax.set_xticks(np.arange(0, self.buffer_seconds + 0.1, 1))
        self.ecg_ax.set_yticks(np.arange(-1.5, 1.6, 0.5))
        self.ecg_ax.grid(which="minor", color="#113322", linewidth=0.4)
        self.ecg_ax.grid(which="major", color="#225544", linewidth=0.8)
        self.ecg_ax.tick_params(labelbottom=False, labelleft=False, length=0)
        for spine in self.ecg_ax.spines.values():
            spine.set_visible(False)
        self.ecg_canvas.draw_idle()
        return estimated_hr

    def draw_resp_wave(self, rate: float) -> None:
        """Scroll the respiration waveform forward by one frame's worth of samples.

        This advances a running phase and rolls one new sample into a persistent buffer,
        exactly like draw_ecg does for the ECG trace. That keeps the wave continuous frame
        to frame instead of restarting from t=0 every single call (which is what caused the
        old "stuck at the edges, pumping in the middle" look).
        """
        dt = 1.0 / self.resp_fs
        freq = rate / 60.0

        self.resp_phase += 2 * np.pi * freq * dt
        new_value = np.sin(self.resp_phase) + 0.12 * np.sin(2 * self.resp_phase + 0.4)

        self.resp_buffer[:-1] = self.resp_buffer[1:]
        self.resp_buffer[-1] = new_value

        self.resp_ax.clear()
        self.resp_ax.set_facecolor("black")
        self.resp_ax.plot(self.resp_x, self.resp_buffer, color=BLUE, linewidth=2)
        self.resp_ax.set_title("Simulated Chest Expansion Waveform from Estimated Breathing Rate", color=TEXT, fontsize=12)
        self.resp_ax.set_xlim(0, self.resp_buffer_seconds)
        self.resp_ax.set_ylim(-1.25, 1.25)
        self.resp_ax.tick_params(labelbottom=False, labelleft=False, length=0)
        for spine in self.resp_ax.spines.values():
            spine.set_visible(False)
        self.resp_canvas.draw_idle()

    def draw_trend(self, df: pd.DataFrame) -> None:
        visible = df.iloc[: self.row_index + 1].copy()
        if visible.empty:
            return

        full_step = max(1, len(df) // 1500)
        visible_step = max(1, len(visible) // 1500)
        full_df = df.iloc[::full_step]
        plot_df = visible.iloc[::visible_step]

        self.trend_ax.clear()
        self.trend_ax.set_facecolor("black")

        # Full 30-day path in the background, live playback on top.
        self.trend_ax.plot(full_df["simulation_day"], full_df["breathing_rate_bpm"], color="#304050", linewidth=0.8, alpha=0.45)
        self.trend_ax.plot(plot_df["simulation_day"], plot_df["breathing_rate_bpm"], color=BLUE, linewidth=1.3)
        self.trend_ax.axhline(12, color=YELLOW, linestyle="--", linewidth=1)
        self.trend_ax.axhline(20, color=YELLOW, linestyle="--", linewidth=1)
        self.trend_ax.set_xlim(0, 30)
        self.trend_ax.set_title("Breathing Frequency During 30-Day Simulation", color=TEXT, fontsize=12)
        self.trend_ax.set_xlabel("Simulation Day", color=TEXT)
        self.trend_ax.set_ylabel("Breaths/min", color=TEXT)
        self.trend_ax.tick_params(colors=TEXT)
        self.trend_ax.grid(True, color="#263443", linewidth=0.5)
        for spine in self.trend_ax.spines.values():
            spine.set_color(GRAY)
        self.trend_canvas.draw_idle()

    def count_sustained_high_episodes(self, df: pd.DataFrame) -> int:
        episodes = 0
        run = 0
        for status in df["status"]:
            if status == "High":
                run += 1
            else:
                if run >= SUSTAINED_HIGH_MINUTES:
                    episodes += 1
                run = 0
        if run >= SUSTAINED_HIGH_MINUTES:
            episodes += 1
        return episodes

    def update_app(self) -> None:
        try:
            if not DATA_FILE.exists():
                self.info.config(text=f"CSV not found:\n{DATA_FILE}\n\nRun simulate.py first.")
                self.root.after(1000, self.update_app)
                return

            df = pd.read_csv(DATA_FILE)
            if df.empty:
                self.info.config(text="CSV exists but has no rows yet.")
                self.root.after(1000, self.update_app)
                return

            latest = df.iloc[self.row_index]
            hr = float(latest["heart_rate_bpm"])
            br = float(latest["breathing_rate_bpm"])
            status = str(latest["status"])

            estimated_hr = self.draw_ecg(hr)
            self.hr_label.config(text=f"{hr:.0f} bpm")
            self.hr_est_label.config(text="-- bpm" if estimated_hr is None else f"{estimated_hr:.0f} bpm")
            self.br_label.config(text=f"{br:.1f} /min")
            self.status.config(text=status.upper(), fg=GREEN if status == "Normal" else YELLOW if status == "Low" else RED)

            current_df = df.iloc[: self.row_index + 1]
            status_counts = current_df["status"].value_counts()
            sustained_high = self.count_sustained_high_episodes(current_df)

            if sustained_high > 0:
                self.alert.config(text=f"ALERT: {sustained_high} sustained high episode(s)", fg=RED)
            else:
                self.alert.config(text="No sustained high alert", fg=GREEN)

            self.info.config(
                text=(
                    f"Simulation Day: {float(latest['simulation_day']):.2f} / 30\n"
                    f"Current Minute: {int(latest['simulation_minute'])}\n\n"
                    f"Average BR so far: {current_df['breathing_rate_bpm'].mean():.2f} /min\n"
                    f"Maximum BR so far: {current_df['breathing_rate_bpm'].max():.2f} /min\n"
                    f"Minimum BR so far: {current_df['breathing_rate_bpm'].min():.2f} /min\n\n"
                    f"Normal readings: {int(status_counts.get('Normal', 0))}\n"
                    f"High readings: {int(status_counts.get('High', 0))}\n"
                    f"Low readings: {int(status_counts.get('Low', 0))}\n"
                    f"Sustained high episodes: {sustained_high}\n\n"
                    "Thresholds:\n"
                    "Low < 12 /min\n"
                    "Normal 12-20 /min\n"
                    "High > 20 /min\n"
                    f"Alert if High >= {SUSTAINED_HIGH_MINUTES} min"
                )
            )

            self.draw_resp_wave(br)
            if self.row_index % 10 == 0:
                self.draw_trend(df)

            self.row_index = (self.row_index + 1) % len(df)

        except Exception as error:
            self.info.config(text=f"Error:\n{error}")

        self.root.after(50, self.update_app)


if __name__ == "__main__":
    root = tk.Tk()
    ECGMonitorApp(root)
    root.mainloop()
