"""Example: encode SSVEP CSV recordings using FHRR.fpe for time-based encoding.

Saves encoded hypervectors per window to recordings/encoded_fpe.npz
"""
import os
import sys
import glob
import math
import numpy as np
import cupy as cp

try:
    from hdc_sanjay.src.algebra import FHRR
except ModuleNotFoundError:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from hdc_sanjay.src.algebra import FHRR


def load_csv_numeric(path: str) -> np.ndarray:
    # simple numeric CSV loader — header + numeric rows
    return np.genfromtxt(path, delimiter=',', skip_header=1)


def windows_from_data(ch_data: np.ndarray, timestamps: np.ndarray, fs: float, win_s: float, step_s: float):
    win_samples = int(round(win_s * fs))
    step_samples = int(round(step_s * fs))
    n = ch_data.shape[0]
    for start in range(0, n - win_samples + 1, step_samples):
        end = start + win_samples
        yield start, end, ch_data[start:end, :]


def harmonic_powers(window: np.ndarray, fs: float, base_freq: float, n_harmonics: int = 3):
    # window: (samples, channels)
    win_len = window.shape[0]
    hann = np.hanning(win_len)
    freqs = np.fft.rfftfreq(win_len, d=1.0 / fs)
    res = []
    for ch in range(window.shape[1]):
        sig = window[:, ch] * hann
        P = np.abs(np.fft.rfft(sig)) ** 2
        vals = []
        for h in range(1, n_harmonics + 1):
            target = base_freq * h
            idx = np.argmin(np.abs(freqs - target))
            vals.append(P[idx])
        res.append(vals)
    return np.array(res)  # shape (channels, n_harmonics)


def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    recordings_dir = os.path.join(project_root, 'recordings')
    csvs = sorted(glob.glob(os.path.join(recordings_dir, 'Track_*.csv')))
    if not csvs:
        print('No CSV recordings found in', recordings_dir)
        return

    fhrr = FHRR(dim=10000)
    D = fhrr.dim

    # encoding bases
    n_channels = 8
    n_harm = 3
    channel_bases = [fhrr.random() for _ in range(n_channels)]
    harmonic_bases = [fhrr.random() for _ in range(n_harm)]
    time_base = fhrr.random()

    encodings = []
    labels = []
    file_index = 0
    for csv in csvs:
        file_index += 1
        print('Processing', os.path.basename(csv))
        data = load_csv_numeric(csv)
        if data.size == 0:
            continue
        timestamps = data[:, 0]
        ch_data = data[:, 1:1 + n_channels]
        marker = data[:, 9] if data.shape[1] > 9 else np.zeros(data.shape[0])

        # estimate fs from timestamps
        dt = np.median(np.diff(timestamps))
        fs = 1.0 / dt if dt > 0 else 250.0
        print(f'  estimated fs={fs:.1f} Hz, samples={ch_data.shape[0]}')

        win_s = 1.0
        step_s = 0.2

        window_idx = 0
        for start, end, win in windows_from_data(ch_data, timestamps, fs, win_s, step_s):
            window_idx += 1
            center_time = timestamps[start + (end - start) // 2]
            # determine base frequency for window (use marker column median)
            f_candidates = marker[start:end]
            if f_candidates.size:
                base_freq = float(np.median(f_candidates))
            else:
                base_freq = 10.0

            powers = harmonic_powers(win, fs, base_freq, n_harmonics=n_harm)
            # normalize powers per-window
            pvec = np.log1p(powers)  # (channels, n_harm)
            pvec = pvec / (np.linalg.norm(pvec) + 1e-12)

            # build encoding using fpe for feature magnitude and time
            time_t = float(center_time - timestamps[0])  # seconds relative to start
            time_hv = fhrr.fpe(time_base, 0.05 * time_t)

            parts = []
            for ch in range(n_channels):
                for h in range(n_harm):
                    # feature hv for harmonic h with magnitude pvec[ch, h]
                    feat = fhrr.fpe(harmonic_bases[h], 0.2 * float(pvec[ch, h]))
                    part = FHRR.bind(channel_bases[ch], feat)
                    part = FHRR.bind(part, time_hv)
                    parts.append(part)

            stacked = cp.stack(parts, axis=0)  # (n_parts, D)
            bundled = fhrr.bundles(stacked)
            centroid = fhrr.normalize(bundled)

            encodings.append(cp.asnumpy(centroid))
            labels.append(base_freq)

        print(f'  windows encoded: {window_idx}')

    encodings = np.stack(encodings)
    labels = np.array(labels)

    out_path = os.path.join(recordings_dir, 'encoded_fpe.npz')
    np.savez_compressed(out_path, encodings=encodings, labels=labels)
    print('Saved encodings to', out_path)

    # show a small summary: cosine similarity between first few encodings
    if encodings.shape[0] >= 2:
        # bring back to cupy for FHRR.cosine_similarity convenience
        a = cp.array(encodings[0])
        b = cp.array(encodings[1])
        sim = FHRR.cosine_similarity(a, b)
        print(f'Cosine similarity between first two encodings: {sim:.4f}')


if __name__ == '__main__':
    main()
