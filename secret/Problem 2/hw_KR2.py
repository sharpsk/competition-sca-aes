#!/usr/bin/env python3
"""
AES Side Channel Analysis Challenge Solver - Problem 1 (Aligned Traces)
- Loads ciphertexts from CSV and aligned traces from NPY
- Performs last-round CPA (invSBox(C ^ k10), HW model) on all 16 bytes
- Saves *ranked* key candidates per byte, with correlation values and ranks
- Plots correlation summary per byte
- Reconstructs AES-128 master key (K0) from recovered round-10 key (K10)
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from typing import Tuple, List

# ---------- AES helpers (S-box + inv S-box + inverse key schedule) ----------

SBOX = np.array([
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

# Compute inverse sbox from sbox
INV_SBOX = np.zeros_like(SBOX)
for i, v in enumerate(SBOX):
    INV_SBOX[v] = i
INV_SBOX = INV_SBOX.astype(np.uint8)

HW = np.array([bin(x).count("1") for x in range(256)], dtype=np.uint8)

def inv_key_schedule_from_k10(k10: List[int]) -> List[List[int]]:
    """Reconstruct K0..K10 from round-10 key (AES-128)."""
    def split_words(key16: List[int]) -> List[List[int]]:
        return [key16[i:i+4] for i in range(0, 16, 4)]
    def merge_words(words: List[List[int]]) -> List[int]:
        return [b for w in words for b in w]
    def rot_word(w: List[int]) -> List[int]:
        return [w[1], w[2], w[3], w[0]]
    def sub_word(w: List[int]) -> List[int]:
        return [int(SBOX[x]) for x in w]

    RCON = [0x00,0x01,0x02,0x04,0x08,0x10,0x20,0x40,0x80,0x1B,0x36]
    W: List[List[int]] = [None] * 44  # type: ignore
    w10 = split_words(k10)
    for j in range(4):
        W[40+j] = w10[j]
    for i in range(43, 3, -1):
        if i % 4 == 0:
            temp = sub_word(rot_word(W[i-1]))
            temp[0] ^= RCON[i // 4]
            W[i-4] = [W[i][k] ^ temp[k] for k in range(4)]
        else:
            W[i-4] = [W[i][k] ^ W[i-1][k] for k in range(4)]
    round_keys: List[List[int]] = []
    for r in range(11):
        round_keys.append(merge_words([W[4*r + j] for j in range(4)]))
    return round_keys

# ---------- Data loading ----------

def load_ciphertexts_and_traces(csv_filename: str, npy_filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    CSV format: plaintext(hex), ciphertext(hex), trace samples...
    We only need the ciphertexts (2nd col) for last-round CPA.
    """
    print(f"Loading ciphertexts from {csv_filename}...")
    df = pd.read_csv(csv_filename, header=None)
    ct = df.iloc[:, 1].apply(lambda x: bytes.fromhex(str(x))).values
    ciphertexts = np.array([list(c) for c in ct], dtype=np.uint8)

    print(f"Loading aligned traces from {npy_filename}...")
    traces = np.load(npy_filename).astype(np.float32)

    # Trim to common count
    n = min(len(traces), len(ciphertexts))
    traces = traces[:n]
    ciphertexts = ciphertexts[:n]
    print(f"Loaded {n} aligned traces with {traces.shape[1]} samples each")
    return ciphertexts, traces

# ---------- CPA core (last round) ----------

def cpa_last_round(traces: np.ndarray, ciphertexts: np.ndarray, byte_pos: int) -> Tuple[int, float, np.ndarray]:
    """
    For a fixed byte position, compute correlations for all 256 key guesses
    using leakage = HW( invSBox(C[byte] ^ k10) ).
    Returns:
      best_key, best_corr_value, correlations[256, num_samples]
    """
    num_traces, num_samples = traces.shape
    correlations = np.zeros((256, num_samples), dtype=np.float32)

    ct_byte = ciphertexts[:, byte_pos]
    # Pre-normalize traces per sample (z-score), improves numerical stability
    y = traces
    y_mean = y.mean(axis=0)
    y_std = y.std(axis=0)
    y_std[y_std == 0] = 1.0
    y_norm = (y - y_mean) / y_std

    for key_guess in range(256):
        inter = INV_SBOX[ct_byte ^ key_guess]
        x = HW[inter].astype(np.float32)
        x_mean = x.mean()
        x_std = x.std()
        if x_std == 0:
            continue
        x_centered = (x - x_mean) / x_std
        # Pearson correlation across all samples at once
        correlations[key_guess, :] = np.abs((y_norm * x_centered[:, None]).sum(axis=0) / (num_traces - 1))

    max_corr_per_guess = correlations.max(axis=1)
    best_key = int(np.argmax(max_corr_per_guess))
    best_corr = float(max_corr_per_guess[best_key])
    return best_key, best_corr, correlations

def rank_candidates(correlations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    max_corrs = correlations.max(axis=1)
    order = np.argsort(max_corrs)[::-1]
    return order, max_corrs

def plot_corr_summary(correlations: np.ndarray, byte_pos: int, outdir: str):
    max_corrs = correlations.max(axis=1)
    plt.figure(figsize=(10, 5))
    plt.plot(range(256), max_corrs, marker="o", markersize=2, linewidth=1)
    plt.title(f"CPA (Last Round) — Byte {byte_pos}")
    plt.xlabel("Key Guess (0–255)")
    plt.ylabel("Max |Correlation|")
    plt.grid(True)
    best = int(np.argmax(max_corrs))
    plt.axvline(best, color="r", linestyle="--", label=f"Best: 0x{best:02X}")
    plt.legend()
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(os.path.join(outdir, f"byte_{byte_pos:02d}_correlation.png"), dpi=150)
    plt.close()

# ---------- Orchestration ----------

def solve_problem1_aligned(csv_filename: str = "real_power_trace.csv",
                           npy_filename: str = "hw_aligned_traces.npy"):
    print("\n=== SOLVING PROBLEM 1 (Aligned Traces): Last-Round CPA ===")
    ciphertexts, traces = load_ciphertexts_and_traces(csv_filename, npy_filename)

    results_dir = "results_problem1_aligned"
    os.makedirs(results_dir, exist_ok=True)

    recovered_k10: List[int] = []

    for byte_pos in range(16):
        print(f"\nAttacking key byte {byte_pos} ...")
        best_key, best_corr, correlations = cpa_last_round(traces, ciphertexts, byte_pos)
        recovered_k10.append(best_key)

        # Rank & save in requested format
        ranks, max_corrs = rank_candidates(correlations)
        print("  Top 5 candidates:")
        for r, k in enumerate(ranks[:5], start=1):
            print(f"    Rank {r:2d}: 0x{k:02X} (corr={max_corrs[k]:.4f})")

        with open(os.path.join(results_dir, f"byte_{byte_pos:02d}_rank.txt"), "w") as f:
            for r, k in enumerate(ranks, start=1):
                f.write(f"Rank {r:3d}: 0x{k:02X} (corr={max_corrs[k]:.6f})\n")

        # Plot per-byte correlation summary
        plot_corr_summary(correlations, byte_pos, results_dir)

    # Compose K10
    k10_hex = "".join(f"{b:02X}" for b in recovered_k10)
    print(f"\nRecovered Round-10 Key (K10): {k10_hex}")

    # Reconstruct K0 (master key)
    round_keys = inv_key_schedule_from_k10(recovered_k10)
    k0_hex = "".join(f"{b:02X}" for b in round_keys[0])
    print(f"Reconstructed AES Master Key (K0): {k0_hex}")

    # Save keys summary + per-round keys
    with open(os.path.join(results_dir, "recovered_keys_summary.txt"), "w") as f:
        f.write(f"K10 (Round 10): {k10_hex}\n")
        f.write(f"K0  (Master) : {k0_hex}\n\n")
        for r, rk in enumerate(round_keys):
            f.write(f"Round {r:02d} Key: {''.join(f'{b:02X}' for b in rk)}\n")

    # Also save each round key separately if you want strict deliverables
    for r, rk in enumerate(round_keys):
        with open(os.path.join(results_dir, f"round_{r:02d}_key.txt"), "w") as f:
            f.write("".join(f"{b:02X}" for b in rk) + "\n")

    print(f"\nAll results saved in: {results_dir}/")
    return recovered_k10, round_keys


if __name__ == "__main__":
    solve_problem1_aligned("real_power_trace.csv", "hw_aligned_traces.npy")
