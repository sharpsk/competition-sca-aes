#!/usr/bin/env python3
"""
Integrated preprocessing + enhanced CPA (10th round)
Full version with AES S-box included.
SVD & Savitzky-Golay removed; baseline correction, alignment, normalization preserved.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, List

# ----- USER-TUNABLE PARAMETERS -----
CSV_PATH = "real_power_trace.csv"
RESULT_DIR = "results_withoutSVD"
os.makedirs(RESULT_DIR, exist_ok=True)

BASELINE_SAMPLES = 100
REMOVE_BAD_TRACES = True
BAD_MEAN_ZTH = 6.0
BAD_STD_ZTH = 6.0
ALIGN_MAX_SHIFT = 50
NORMALIZE_COLUMNWISE = True

POI_TOP_SAMPLES = 60
REFINE_TOP_K = 6
REFINE_POI_SAMPLES = 200
USE_REFINEMENT = True

PLOT_SAMPLE_IDX = 0
PLOT_NOVERLAY = 8

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
    if isinstance(val, bytes): return val
    if isinstance(val, str): return bytes.fromhex(val.strip().replace(" ", ""))
    raise ValueError(f"Unexpected type {type(val)}")

def load_and_parse_csv(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(path, header=None, dtype=object)
    plaintexts = np.vstack([np.frombuffer(clean_hex(x), dtype=np.uint8) for x in df.iloc[:,0]])
    ciphertexts = np.vstack([np.frombuffer(clean_hex(x), dtype=np.uint8) for x in df.iloc[:,1]])
    traces = df.iloc[:,2:].apply(pd.to_numeric, errors='coerce').values.astype(np.float64)
    return plaintexts, ciphertexts, traces

def detect_bad_traces(traces: np.ndarray, mean_z_thresh=BAD_MEAN_ZTH, std_z_thresh=BAD_STD_ZTH):
    means = np.mean(traces, axis=1); stds = np.std(traces, axis=1)
    bad = (stds==0) | np.isnan(stds)
    mean_z = (means - means.mean()) / (means.std()+1e-12)
    std_z = (stds - stds.mean()) / (stds.std()+1e-12)
    bad |= np.abs(mean_z) > mean_z_thresh
    bad |= np.abs(std_z) > std_z_thresh
    keep_mask = ~bad; bad_indices = np.where(bad)[0].tolist()
    return keep_mask, bad_indices

def baseline_correction(traces: np.ndarray, baseline_samples=BASELINE_SAMPLES):
    baseline = np.mean(traces[:, :baseline_samples], axis=1, keepdims=True)
    return traces - baseline

def align_traces_by_template(traces: np.ndarray, ref_trace=None, max_shift=ALIGN_MAX_SHIFT):
    if ref_trace is None: ref_trace = np.median(traces, axis=0)
    aligned = np.empty_like(traces); L = traces.shape[1]
    for i, t in enumerate(traces):
        t0 = t - t.mean(); r0 = ref_trace - ref_trace.mean()
        corr = np.correlate(t0, r0, mode='full')
        shift = int(corr.argmax() - (L - 1)); shift = max(-max_shift, min(max_shift, shift))
        rolled = np.roll(t, -shift)
        if shift > 0: rolled[-shift:] = np.mean(t[:min(50,L)])
        elif shift < 0: rolled[: -shift] = np.mean(t[:min(50,L)])
        aligned[i] = rolled
    return aligned

def normalize_columnwise(traces: np.ndarray):
    mu = np.mean(traces, axis=0, keepdims=True)
    sigma = np.std(traces, axis=0, keepdims=True) + 1e-12
    return (traces - mu)/sigma

def pearson_correlation(hyp: np.ndarray, traces: np.ndarray) -> np.ndarray:
    hyp, traces = hyp.astype(np.float64), traces.astype(np.float64)
    hyp_dev = hyp - hyp.mean(axis=0, keepdims=True)
    traces_dev = traces - traces.mean(axis=0, keepdims=True)
    numerator = hyp_dev.T @ traces_dev
    denom = np.sqrt(np.sum(hyp_dev**2, axis=0)[:,None] * np.sum(traces_dev**2, axis=0))
    return numerator / (denom + 1e-12)

# ----- Preprocessing pipeline -----
def preprocess_pipeline(csv_path: str):
    pts, cts, traces = load_and_parse_csv(csv_path)
    n_traces, n_samples = traces.shape
    print(f"Loaded {n_traces} traces × {n_samples} samples")
    preview_n = min(PLOT_NOVERLAY, n_traces)
    plt.figure(figsize=(10,4))
    for i in range(preview_n): plt.plot(traces[i], alpha=0.6)
    plt.title("Raw traces overlay"); plt.xlabel("Sample"); plt.ylabel("Amplitude")
    plt.tight_layout(); plt.savefig(os.path.join(RESULT_DIR,"raw_overlay.png")); plt.close()
    keep_mask, bad_indices = detect_bad_traces(traces)
    if REMOVE_BAD_TRACES and len(bad_indices)>0:
        traces, pts, cts = traces[keep_mask], pts[keep_mask], cts[keep_mask]
    traces_base = baseline_correction(traces)
    traces_aligned = align_traces_by_template(traces_base)
    if NORMALIZE_COLUMNWISE: traces_proc = normalize_columnwise(traces_aligned)
    else: traces_proc = traces_aligned
    return pts, cts, traces_proc, keep_mask, bad_indices

# ----- CPA functions -----
def select_poi_by_maxcorr_per_sample(corr_matrix: np.ndarray, top_n:int) -> np.ndarray:
    max_per_sample = np.max(np.abs(corr_matrix), axis=0)
    idx_sorted = np.argsort(-max_per_sample)
    return np.sort(idx_sorted[:top_n])

def multi_sample_score(corr_matrix: np.ndarray, poi_indices: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum(np.abs(corr_matrix[:, poi_indices])**2, axis=1))

def reconstruct_round_keys_from_k10(k10: List[int]) -> List[List[int]]:
    RCON = [0x00,0x01,0x02,0x04,0x08,0x10,0x20,0x40,0x80,0x1B,0x36]
    W: List[List[int]] = [None]*44
    words10 = [list(k10[i:i+4]) for i in range(0,16,4)]
    for j in range(4): W[40+j] = words10[j]
    def rot_word(w): return [w[1],w[2],w[3],w[0]]
    def sub_word(w): return [int(sbox[b]) for b in w]
    for i in range(43,3,-1):
        if i%4==0:
            temp = sub_word(rot_word(W[i-1]))
            temp[0] ^= RCON[i//4]
            W[i-4] = [W[i][k]^temp[k] for k in range(4)]
        else:
            W[i-4] = [W[i][k]^W[i-1][k] for k in range(4)]
    round_keys = []
    for r in range(11):
        rk = [W[4*r+j] for j in range(4)]
        round_keys.append([b for w in rk for b in w])
    return round_keys

def run_cpa_10th_round_enhanced_and_save(plaintexts, ciphertexts, traces, result_dir=RESULT_DIR):
    N,S = traces.shape
    recovered_key = np.zeros(16, dtype=np.uint8)
    summary_lines = []

    for byte_pos in range(16):
        ct_byte = ciphertexts[:, byte_pos][:,None].astype(np.int32)
        keys = np.arange(256)[None,:]
        intermediates = inv_sbox[np.bitwise_xor(ct_byte, keys)]
        hyp = hw[intermediates]
        corr_full = pearson_correlation(hyp, traces)
        poi = select_poi_by_maxcorr_per_sample(corr_full, POI_TOP_SAMPLES)
        scores = multi_sample_score(corr_full, poi)
        best_k = int(np.argmax(scores))
        recovered_key[byte_pos] = best_k
        ranking_order = np.argsort(-scores)
        with open(os.path.join(result_dir,f"byte_{byte_pos:02d}_rank.txt"),"w") as fh:
            for rank_idx,key_guess in enumerate(ranking_order,1):
                fh.write(f"{rank_idx:3d} | 0x{key_guess:02x} | {scores[key_guess]:.8e}\n")
        top5 = ranking_order[:5]
        top5_str = ", ".join(f"0x{k:02x}({scores[k]:.3e})" for k in top5)
        summary_lines.append(f"Byte {byte_pos:02d} | best=0x{recovered_key[byte_pos]:02x} | top5=[{top5_str}]")
    with open(os.path.join(result_dir,"recovered_key.txt"),"w") as f:
        f.write(" ".join(f"{b:02x}" for b in recovered_key)+"\n")
    with open(os.path.join(result_dir,"correlation_ranks.txt"),"w") as f:
        for line in summary_lines: f.write(line+"\n")
    round_keys = reconstruct_round_keys_from_k10(recovered_key.tolist())
    with open(os.path.join(result_dir,"master_key.txt"),"w") as f:
        f.write("".join(f"{b:02x}" for b in round_keys[0])+"\n")
    for r,rk in enumerate(round_keys):
        with open(os.path.join(result_dir,f"round_{r:02d}_key.txt"),"w") as f:
            f.write("".join(f"{b:02x}" for b in rk)+"\n")
    print("Saved recovered_key.txt, master_key.txt and per-byte rank files in", result_dir)
    return recovered_key

# ----- Main -----
def main():
    print("Loading and preprocessing...")
    plaintexts, ciphertexts, traces_proc, keep_mask, bad_idx = preprocess_pipeline(CSV_PATH)
    print("Running enhanced CPA (10th round)...")
    recovered_key = run_cpa_10th_round_enhanced_and_save(plaintexts, ciphertexts, traces_proc, result_dir=RESULT_DIR)
    print("Recovered 10th-round key (hex):", " ".join(f"{b:02x}" for b in recovered_key))

if __name__=="__main__":
    main()
