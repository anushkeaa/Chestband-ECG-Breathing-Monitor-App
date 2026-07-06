# Chestband ECG & Breathing Monitor

Simulated chestband project: 30 days of heart-rate + breathing data, an analysis
script, and a live tkinter dashboard.

## Files

- `simulate.py` — generates 30 days of simulated data → `data/chestband_data.csv`
- `analyze.py` — reads that CSV, writes a report + 2 charts to `data/`
- `monitor.py` — live tkinter dashboard (ECG wave, breathing wave, trend chart)
- `notebook.ipynb` — same pipeline, notebook form, with a fast bounded playback preview

## How to run

```
python simulate.py     # 1. creates data/chestband_data.csv (takes a bit, ~43,200 minutes)
python analyze.py      # 2. creates report + charts in data/
python monitor.py      # 3. opens the live dashboard window
```

Or open `notebook.ipynb` and run all cells top to bottom.

## Notes

- Everything reads/writes `data/`, next to whichever script you run. No assumed parent
  folder, no path guessing.
- `monitor.py`'s breathing wave now scrolls continuously (see comment in `draw_resp_wave`)
  instead of resetting to a fresh curve every frame — that resetting was the "harmonium"
  bug where the edges looked fixed and the middle just pumped in and out.
