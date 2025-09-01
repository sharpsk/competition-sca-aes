#!/usr/bin/env python3
"""
Integrated preprocessing + enhanced CPA (10th round)
Extended to save results in Code-2-style output files:
- results/byte_XX_rank.txt    (256 candidates, sorted)
- results/correlation_ranks.txt  (summary)
- results/recovered_key.txt   (round-10 key)
- results/master_key.txt      (AES-128 master key K0)
- results/round_00_key.txt ... round_10_key.txt
Diagnostic plots from original Code 1 are preserved.
Usage: python key_r10_prepro_final_with_results.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, List

# ----- USER-TUNABLE PARAMETERS -----
CSV_PATH = "real_power_trace.csv"
RESULT_DIR = "results_withSVD"   # fixed results folder as requested
os.makedirs(RESULT_DIR, exist_ok=True)

# Preprocessing params
BASELINE_SAMPLES = 100       # number of initial samples used for baseline subtraction
REMOVE_BAD_TRACES = True
BAD_MEAN_ZTH = 6.0
BAD_STD_ZTH = 6.0
ALIGN_MAX_SHIFT = 50        # max shift (samples) when aligning traces via xcorr
SAVGOL_WINDOW = 21          # odd; Savitzky-Golay smoothing window
SAVGOL_POLY = 3
SVD_ENERGY = 0.90           # fraction of energy to keep in SVD denoising
SVD_MAXCOMP = 200
NORMALIZE_COLUMNWISE = True  # z-score per sample (column) after preprocessing

# CPA tuning
POI_TOP_SAMPLES = 60        # number of POI samples to select per byte (e.g., 30..200)
REFINE_TOP_K = 6            # number of top key guesses to refine in 2nd pass
REFINE_POI_SAMPLES = 200    # POI size for refinement (>= POI_TOP_SAMPLES)
USE_REFINEMENT = True       # whether to run the 2-pass refine step

# Plotting params
PLOT_SAMPLE_IDX = 0         # example trace index (after removing bad traces) for preprocessing-stage plots
PLOT_NOVERLAY = 8           # number of traces to overlay for raw/examined plots

# ----- AES tables -----
sbox = np.array([
    0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
    0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
    0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
    0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
    0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
    0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
    0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
    0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
    0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
    0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
    0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
    0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
    0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
    0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
    0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
    0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16
], dtype=np.uint8)

inv_sbox = np.zeros(256, dtype=np.uint8)
for i in range(256):
    inv_sbox[sbox[i]] = i

hw = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)

# ----- Helper functions -----
def clean_hex(val):
    """Accepts str or bytes and returns raw bytes of length 16."""
    if isinstance(val, bytes):
        return val
    if isinstance(val, str):
        v = val.strip().replace(" ", "").upper()
        return bytes.fromhex(v)
    raise ValueError(f"Unexpected plaintext/ciphertext type: {type(val)}")

def load_and_parse_csv(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (plaintexts (N,16) uint8, ciphertexts (N,16) uint8, traces (N,S) float64)
    CSV layout expected: col0=plaintext(hex16), col1=ciphertext(hex16), col2.. = trace samples
    """
    df = pd.read_csv(path, header=None, dtype=object)
    plaintexts = np.vstack([np.frombuffer(clean_hex(x), dtype=np.uint8) for x in df.iloc[:, 0]])
    ciphertexts = np.vstack([np.frombuffer(clean_hex(x), dtype=np.uint8) for x in df.iloc[:, 1]])
    traces = df.iloc[:, 2:].apply(pd.to_numeric, errors='coerce').values.astype(np.float64)
    return plaintexts, ciphertexts, traces

def detect_bad_traces(traces: np.ndarray, mean_z_thresh=BAD_MEAN_ZTH, std_z_thresh=BAD_STD_ZTH):
    means = np.mean(traces, axis=1)
    stds = np.std(traces, axis=1)
    bad = np.zeros(traces.shape[0], dtype=bool)
    bad |= np.isnan(stds)
    bad |= (stds == 0)
    mean_z = (means - means.mean()) / (means.std() + 1e-12)
    std_z  = (stds  - stds.mean())  / (stds.std()  + 1e-12)
    bad |= (np.abs(mean_z) > mean_z_thresh)
    bad |= (np.abs(std_z)  > std_z_thresh)
    keep_mask = ~bad
    bad_indices = np.where(bad)[0].tolist()
    return keep_mask, bad_indices

def baseline_correction(traces: np.ndarray, baseline_samples: int = BASELINE_SAMPLES):
    baseline = np.mean(traces[:, :baseline_samples], axis=1, keepdims=True)
    return traces - baseline

def align_traces_by_template(traces: np.ndarray, ref_trace=None, max_shift: int = ALIGN_MAX_SHIFT):
    """
    Align every trace to ref_trace using cross-correlation (simple roll).
    Caps shifts to +/- max_shift.
    """
    if ref_trace is None:
        ref_trace = np.median(traces, axis=0)
    aligned = np.empty_like(traces)
    L = traces.shape[1]
    for i, t in enumerate(traces):
        t0 = t - t.mean()
        r0 = ref_trace - ref_trace.mean()
        corr = np.correlate(t0, r0, mode='full')
        shift = int(corr.argmax() - (L - 1))
        shift = max(-max_shift, min(max_shift, shift))
        rolled = np.roll(t, -shift)
        if shift > 0:
            rolled[-shift:] = np.mean(t[:min(50, L)])
        elif shift < 0:
            rolled[: -shift] = np.mean(t[:min(50, L)])
        aligned[i] = rolled
    return aligned

def smooth_traces_savgol(traces: np.ndarray, window=SAVGOL_WINDOW, polyorder=SAVGOL_POLY):
    try:
        from scipy.signal import savgol_filter
        if window % 2 == 0:
            window += 1
        if window >= traces.shape[1]:
            window = traces.shape[1] - 1
            if window % 2 == 0:
                window -= 1
        return savgol_filter(traces, window_length=window, polyorder=polyorder, axis=1)
    except Exception:
        w = 11
        kernel = np.ones(w) / w
        return np.array([np.convolve(t, kernel, mode='same') for t in traces])

def svd_denoise(traces: np.ndarray, energy_threshold=SVD_ENERGY, max_components=SVD_MAXCOMP):
    mean = np.mean(traces, axis=0, keepdims=True)
    X = traces - mean
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    energy = np.cumsum(S**2)
    energy /= (energy[-1] + 1e-20)
    k = int(np.searchsorted(energy, energy_threshold) + 1)
    k = max(1, min(k, max_components, len(S)))
    Xk = (U[:, :k] * S[:k]) @ Vt[:k, :]
    return Xk + mean, k

def normalize_columnwise(traces: np.ndarray):
    mu = np.mean(traces, axis=0, keepdims=True)
    sigma = np.std(traces, axis=0, keepdims=True) + 1e-10
    return (traces - mu) / sigma

# Correct Pearson correlation: hyp shape (N, K), traces shape (N, S) -> returns (K, S)
def pearson_correlation(hyp: np.ndarray, traces: np.ndarray) -> np.ndarray:
    hyp = hyp.astype(np.float64)
    traces = traces.astype(np.float64)
    hyp_mean = np.mean(hyp, axis=0, keepdims=True)       # (1, K)
    traces_mean = np.mean(traces, axis=0, keepdims=True) # (1, S)
    hyp_dev = hyp - hyp_mean                              # (N, K)
    traces_dev = traces - traces_mean                     # (N, S)
    numerator = hyp_dev.T @ traces_dev                    # (K, S)
    hyp_var = np.sum(hyp_dev**2, axis=0)                  # (K,)
    trace_var = np.sum(traces_dev**2, axis=0)             # (S,)
    denom = np.sqrt(np.outer(hyp_var, trace_var))         # (K, S)
    corr = numerator / (denom + 1e-12)
    return corr

# ----- Preprocessing pipeline -----
def preprocess_pipeline(csv_path: str):
    pts, cts, traces = load_and_parse_csv(csv_path)
    n_traces, n_samples = traces.shape
    print(f"Loaded: {n_traces} traces × {n_samples} samples")

    # raw overlay plot
    preview_n = min(PLOT_NOVERLAY, n_traces)
    plt.figure(figsize=(10,4))
    for i in range(preview_n):
        plt.plot(traces[i], alpha=0.6)
    plt.title("Overlay of first traces (raw)")
    plt.xlabel("Sample index"); plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "raw_overlay.png")); plt.close()

    # detect bad traces
    keep_mask, bad_indices = detect_bad_traces(traces)
    print(f"Detected {len(bad_indices)} bad traces (indices): {bad_indices[:30]}")
    if REMOVE_BAD_TRACES and len(bad_indices) > 0:
        traces = traces[keep_mask]
        pts = pts[keep_mask]
        cts = cts[keep_mask]
        print(f"Removed bad traces: now {traces.shape[0]} traces remain")
        with open(os.path.join(RESULT_DIR, "bad_indices.txt"), "w") as f:
            f.write(repr(bad_indices))

    # baseline correction
    traces_base = baseline_correction(traces, baseline_samples=BASELINE_SAMPLES)
    plt.figure(figsize=(10,4))
    plt.plot(traces_base[PLOT_SAMPLE_IDX])
    plt.title("Example trace after baseline correction")
    plt.xlabel("Sample index"); plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "baseline_corrected_example.png")); plt.close()

    # alignment
    traces_aligned = align_traces_by_template(traces_base, max_shift=ALIGN_MAX_SHIFT)
    plt.figure(figsize=(10,4))
    plt.plot(traces_aligned[PLOT_SAMPLE_IDX])
    plt.title("Example trace after alignment")
    plt.xlabel("Sample index"); plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "aligned_example.png")); plt.close()

    # smoothing
    traces_smoothed = smooth_traces_savgol(traces_aligned, window=SAVGOL_WINDOW, polyorder=SAVGOL_POLY)
    plt.figure(figsize=(10,4))
    plt.plot(traces_smoothed[PLOT_SAMPLE_IDX])
    plt.title("Example trace after smoothing (Savitzky-Golay)")
    plt.xlabel("Sample index"); plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "smoothed_example.png")); plt.close()

    # svd denoise
    traces_svd, used_k = svd_denoise(traces_smoothed, energy_threshold=SVD_ENERGY, max_components=SVD_MAXCOMP)
    plt.figure(figsize=(10,4))
    plt.plot(traces_svd[PLOT_SAMPLE_IDX])
    plt.title(f"Example trace after SVD denoising (k={used_k})")
    plt.xlabel("Sample index"); plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "svd_denoised_example.png")); plt.close()

    # combined stages plot
    plt.figure(figsize=(12,5))
    plt.plot(traces[PLOT_SAMPLE_IDX], label="raw", alpha=0.6)
    plt.plot(traces_base[PLOT_SAMPLE_IDX], label="baseline", alpha=0.6)
    plt.plot(traces_aligned[PLOT_SAMPLE_IDX], label="aligned", alpha=0.6)
    plt.plot(traces_smoothed[PLOT_SAMPLE_IDX], label="smoothed", alpha=0.8)
    plt.plot(traces_svd[PLOT_SAMPLE_IDX], label="svd-denoised", alpha=0.9)
    plt.legend()
    plt.title("Preprocessing stages (example trace)")
    plt.xlabel("Sample index"); plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "preprocessing_stages_example.png")); plt.close()

    # normalization columnwise
    if NORMALIZE_COLUMNWISE:
        traces_proc = normalize_columnwise(traces_svd)
    else:
        traces_proc = traces_svd

    with open(os.path.join(RESULT_DIR, "preproc_report.txt"), "w") as fh:
        fh.write(f"Original traces: {n_traces}, samples: {n_samples}\n")
        fh.write(f"Bad indices (detected): {bad_indices}\n")
        fh.write(f"Used SVD components: {used_k}\n")
        fh.write(f"Params: BASELINE_SAMPLES={BASELINE_SAMPLES}, SAVGOL_WINDOW={SAVGOL_WINDOW}, SVD_ENERGY={SVD_ENERGY}\n")

    return pts, cts, traces_proc, keep_mask, bad_indices

# ----- Enhanced CPA (no heatmaps) with Code-2 style saving -----
def select_poi_by_maxcorr_per_sample(corr_matrix: np.ndarray, top_n:int) -> np.ndarray:
    max_per_sample = np.max(np.abs(corr_matrix), axis=0)
    idx_sorted = np.argsort(-max_per_sample)
    poi = np.sort(idx_sorted[:top_n])
    return poi

def multi_sample_score(corr_matrix: np.ndarray, poi_indices: np.ndarray) -> np.ndarray:
    sub = corr_matrix[:, poi_indices]  # (256, P)
    rss = np.sqrt(np.sum((np.abs(sub))**2, axis=1))
    return rss

def reconstruct_round_keys_from_k10(k10: List[int]) -> List[List[int]]:
    """Reconstruct all round keys K0..K10 from round-10 key using inverse AES-128 schedule."""
    RCON = [0x00, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1B, 0x36]

    W: List[List[int]] = [None] * 44
    words10 = [list(k10[i:i+4]) for i in range(0,16,4)]
    for j in range(4):
        W[40 + j] = words10[j]

    def rot_word(w):
        return [w[1], w[2], w[3], w[0]]

    def sub_word(w):
        return [int(sbox[b]) for b in w]

    for i in range(43, 3, -1):
        if i % 4 == 0:
            temp = sub_word(rot_word(W[i - 1]))
            temp[0] ^= RCON[i // 4]
            W[i - 4] = [W[i][k] ^ temp[k] for k in range(4)]
        else:
            W[i - 4] = [W[i][k] ^ W[i - 1][k] for k in range(4)]

    round_keys: List[List[int]] = []
    for r in range(11):
        rk = [W[4 * r + j] for j in range(4)]
        rk_flat = [b for w in rk for b in w]
        round_keys.append(rk_flat)
    return round_keys

def run_cpa_10th_round_enhanced_and_save(plaintexts, ciphertexts, traces, result_dir=RESULT_DIR,
                                         poi_top_samples=POI_TOP_SAMPLES, refine_top_k=REFINE_TOP_K,
                                         refine_poi=REFINE_POI_SAMPLES, use_refinement=USE_REFINEMENT):
    N, S = traces.shape
    assert ciphertexts.shape[0] == N
    recovered_key = np.zeros(16, dtype=np.uint8)

    # Prepare summary file
    summary_lines = []
    for byte_pos in range(16):
        print(f"[enhanced CPA] Attacking byte {byte_pos} ...")
        ct_byte = ciphertexts[:, byte_pos][:, np.newaxis].astype(np.int32)   # (N,1)
        keys = np.arange(256)[np.newaxis, :]                                # (1,256)
        xored = np.bitwise_xor(ct_byte, keys)                               # (N,256)
        intermediates = inv_sbox[xored]                                     # (N,256)
        hyp = hw[intermediates]                                             # (N,256)

        corr_full = pearson_correlation(hyp, traces)                         # (256, S)

        # save per-sample max |corr|
        max_per_sample = np.max(np.abs(corr_full), axis=0)
        plt.figure(figsize=(10,3))
        plt.plot(max_per_sample)
        plt.title(f"Byte {byte_pos:02d} per-sample max |corr|")
        plt.xlabel("Sample index"); plt.ylabel("max |corr|")
        plt.tight_layout()
        plt.savefig(os.path.join(result_dir, f"byte_{byte_pos:02d}_maxper_sample.png"))
        plt.close()

        poi = select_poi_by_maxcorr_per_sample(corr_full, poi_top_samples)
        with open(os.path.join(result_dir, f"byte_{byte_pos:02d}_poi_indices.txt"), "w") as f:
            f.write(",".join(map(str, poi.tolist())))

        # primary scores (RSS over selected POI)
        scores = multi_sample_score(corr_full, poi)   # (256,)
        best_k = int(np.argmax(scores))
        recovered_key[byte_pos] = best_k

        # Save ranking plot (score vs key)
        plt.figure(figsize=(8,3))
        plt.plot(scores)
        plt.title(f"Byte {byte_pos:02d} score (RSS over POI)")
        plt.xlabel("Key guess (0..255)")
        plt.ylabel("Score")
        plt.tight_layout()
        plt.savefig(os.path.join(result_dir, f"byte_{byte_pos:02d}_score_poi{len(poi)}.png"))
        plt.close()

        # Save full ranked candidates (all 256) using 'scores' metric
        ranking_order = np.argsort(-scores)  # descending
        rank_filename = os.path.join(result_dir, f"byte_{byte_pos:02d}_rank.txt")
        with open(rank_filename, "w") as fh:
            fh.write("Rank | Key Guess (hex) | Score(RSS_over_POI)\n")
            fh.write("---- | --------------- | -------------------\n")
            for rank_idx, key_guess in enumerate(ranking_order, start=1):
                fh.write(f"{rank_idx:3d} | 0x{key_guess:02x}           | {scores[key_guess]:.8e}\n")

        print(f"Byte {byte_pos}: top guess initial = 0x{best_k:02x}")

        # Refinement (2nd pass) using larger POI and only top K candidates
        refined_best_k = best_k
        if use_refinement:
            candidate_keys = np.argsort(-scores)[:refine_top_k]
            refine_poi_indices = select_poi_by_maxcorr_per_sample(corr_full, refine_poi)
            refined_scores = []
            for k in candidate_keys:
                row = corr_full[k, refine_poi_indices]  # (P,)
                refined_scores.append(np.sqrt(np.sum(np.abs(row)**2)))
            refined_scores = np.array(refined_scores)
            refined_best_idx = int(np.argmax(refined_scores))
            refined_best_k = int(candidate_keys[refined_best_idx])
            if refined_best_k != best_k:
                print(f"Byte {byte_pos}: refined guess -> 0x{refined_best_k:02x} (was 0x{best_k:02x})")
                recovered_key[byte_pos] = refined_best_k
            else:
                print(f"Byte {byte_pos}: refinement kept 0x{best_k:02x}")

            # Save refinement candidate list
            with open(os.path.join(result_dir, f"byte_{byte_pos:02d}_refine_candidates.txt"), "w") as f:
                f.write(",".join(f"{k:02x}" for k in candidate_keys) + "\n")
                f.write("refined_best," + f"{refined_best_k:02x}\n")

        # For summary: top-5 candidates with scores
        top5 = ranking_order[:5]
        top5_str = ", ".join(f"0x{k:02x}({scores[k]:.3e})" for k in top5)
        summary_lines.append(f"Byte {byte_pos:02d} | best=0x{recovered_key[byte_pos]:02x} | top5=[{top5_str}]")

    # Save recovered round-10 key (space-separated hex bytes) - matching Code 2 style
    with open(os.path.join(result_dir, "recovered_key.txt"), "w") as f:
        f.write(" ".join(f"{b:02x}" for b in recovered_key) + "\n")

    # Save summary file (correlation_ranks.txt)
    with open(os.path.join(result_dir, "correlation_ranks.txt"), "w") as f:
        f.write("Enhanced CPA (RSS over POI) summary per byte:\n")
        for line in summary_lines:
            f.write(line + "\n")

    # Reconstruct AES master key using inverse key schedule and save all round keys
    k10_list = [int(b) for b in recovered_key.tolist()]
    round_keys = reconstruct_round_keys_from_k10(k10_list)
    # round_keys[0] is K0 (master), round_keys[10] is K10
    k0_hex = "".join(f"{b:02x}" for b in round_keys[0])
    with open(os.path.join(result_dir, "master_key.txt"), "w") as f:
        f.write(k0_hex + "\n")

    # Save each round key file
    for r, rk in enumerate(round_keys):
        rk_hex = "".join(f"{b:02x}" for b in rk)
        with open(os.path.join(result_dir, f"round_{r:02d}_key.txt"), "w") as f:
            f.write(rk_hex + "\n")

    print("\nSaved recovered_key.txt, master_key.txt and per-byte rank files in", result_dir)
    return recovered_key

# ----- Main -----
def main():
    print("Loading and preprocessing...")
    plaintexts, ciphertexts, traces_proc, keep_mask, bad_idx = preprocess_pipeline(CSV_PATH)

    print("Running enhanced CPA (10th round) on preprocessed traces and saving results ...")
    recovered_key = run_cpa_10th_round_enhanced_and_save(plaintexts, ciphertexts, traces_proc, result_dir=RESULT_DIR)

    print("\nRecovered 10th-round key (hex):")
    print(" ".join(f"{b:02x}" for b in recovered_key))
    print(f"All results and plots saved in ./{RESULT_DIR}/")

if __name__ == "__main__":
    main()
