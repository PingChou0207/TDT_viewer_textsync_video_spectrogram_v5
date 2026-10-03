# TDT Viewer TextSync LFP Spectrogram v4

A read-only desktop viewer for TDT blocks and synchronized text/CSV traces. It displays epoch events, multi-channel LFP and MU traces, imported TXT traces, and independent LFP and TXT spectrograms on a shared time axis.

## Features

- Six resizable panels: Epoch, LFP, LFP spectrogram, MU, TXT trace, and TXT spectrogram
- Import numeric `.txt` or `.csv` data, with one sample per row and one channel per column; a single column is also supported
- Adjustable TXT sampling rate, amplitude, scale bar, channels, and panel position
- Separate LFP and TXT spectrogram controls and channel selection
- Real and percentage power; constant or linear detrending; optional NeuroExplorer-compatible Gaussian frequency smoothing
- Configurable frequency range, resolution, time step, overlap, color range, and `viridis`/`jet` color maps
- Synchronized cursor and event navigation, image export, session save/open, and gain-normalized TDT traces

TDT blocks are read without modifying the source data. Only open `.tdtv` session files from trusted sources because sessions use Python pickle.

## Run from source

Python 3.12 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install -r requirements-build.txt
python src/tdt_viewer_textsync_lfp_spectrogram_v4.py
```

Click **Open Block** to load a TDT block. Click **Open Text** to add a numeric TXT/CSV trace, then set **TXT sampling rate** to the actual sampling frequency. LFP and TXT spectrogram settings are independent.

## Build desktop applications

- macOS Apple Silicon: run `./build_macos.command`. The generated app and ZIP are in `dist/`. The app is ad-hoc signed but not Apple-notarized; on first launch, Control-click and choose **Open**.
- Windows x64: on a Windows 10/11 computer with Python 3.12, run `build_windows.bat`. Keep the resulting EXE and `_internal` folder together.
- GitHub Actions: run **Build desktop apps** manually, or push a version tag beginning with `v`. The workflow produces downloadable build artifacts for both platforms.

See [build and launch notes](README_BUILD.txt) for file names and paths. The source ZIP from GitHub's **Code → Download ZIP** is not a prebuilt application.

## Credits

Application design and development: **PingChou**. This viewer uses the TDT Python SDK, Python, NumPy, SciPy, PySide6, pyqtgraph, and PyInstaller. TDT and NeuroExplorer are product names or trademarks of their respective owners; this is an independent application.

## License

No open-source license has been granted yet. The source is publicly visible for inspection; copyright remains with the author unless a license is added later.
