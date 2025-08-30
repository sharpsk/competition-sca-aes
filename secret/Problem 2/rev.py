#!/usr/bin/env python3
"""
AES Side Channel Analysis Challenge Solver - Problem 1 (Aligned Traces)
Revised: robust CPA, auto-orient traces, diagnostics, shiftrows helper.
"""
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional

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
    This function will attempt to auto-orient traces (transpose) if shapes mismatch.
    """
    print(f"Loading ciphertexts from {csv_filename}...")
    df = pd.read_csv(csv_filename, header=None)
    ct = df.iloc[:, 1].apply(lambda x: bytes.fromhex(str(x))).values
    ciphertexts = np.array([list(c) for c in ct], dtype=np.uint8)

    print(f"Loading aligned traces from {npy_filename}...")
    traces = np.load(npy_filename)
    traces = traces.astype(np.float32)

    # If traces axis doesn't match number of ciphertexts, try transpose
    if traces.shape[0] != len(ciphertexts) and traces.shape[1] == len(ciphertexts):
        print("Note: transposing traces (shape mismatch detected).")
        traces = traces.T
    if traces.shape[0] != len(ciphertexts):
        raise ValueError(f"Number of traces ({traces.shape[0]}) does not match number of ciphertexts ({len(ciphertexts)}).")

    n = min(len(traces), len(ciphertexts))
    traces = traces[:n]
    ciphertexts = ciphertexts[:n]
    print(f"Loaded {n} aligned traces with {traces.shape[1]} samples each (traces.shape={traces.shape})")
    return ciphertexts, traces

# ---------- CPA core (last round) ----------

def cpa_last_round(traces: np.ndarray, ciphertexts: np.ndarray, byte_pos: int,
                   use_inv_sbox: bool = True) -> Tuple[int, float, np.ndarray]:
    """
    Robust vectorized Pearson correlation:
      correlations[key_guess, time_index] = corr( HW(invSBox(ct_byte ^ key_guess)), trace[:, time_index] )
    Returns best_key, best_corr, correlations (shape: 256 x num_samples)
    """
    # ensure float64 for stable numeric ops
    y = traces.astype(np.float64)
    num_traces, num_samples = y.shape

    # center y per time point
    y_mean = y.mean(axis=0)
    y_centered = y - y_mean
    y_norm_den = np.sqrt((y_centered ** 2).sum(axis=0))
    # avoid divide-by-zero
    y_norm_den[y_norm_den == 0.0] = 1.0

    correlations = np.zeros((256, num_samples), dtype=np.float64)
    ct_byte = ciphertexts[:, byte_pos].astype(np.uint8)

    # choose sbox mapping for hypothesis
    sbox_map = INV_SBOX if use_inv_sbox else SBOX

    for key_guess in range(256):
        # vectorized: inter per trace
        inter = sbox_map[np.bitwise_xor(ct_byte, key_guess)]
        x = HW[inter].astype(np.float64)
        x_centered = x - x.mean()
        x_den = np.sqrt((x_centered ** 2).sum())
        if x_den == 0.0:
            continue
        # numerator: dot product between x_centered and each column (time) of y_centered
        num = x_centered @ y_centered  # shape: (num_samples,)
        correlations[key_guess, :] = num / (x_den * y_norm_den)

    abs_corrs = np.abs(correlations)
    max_corr_per_guess = abs_corrs.max(axis=1)
    best_key = int(np.argmax(max_corr_per_guess))
    best_corr = float(max_corr_per_guess[best_key])
    return best_key, best_corr, correlations

def rank_candidates(correlations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    max_corrs = np.abs(correlations).max(axis=1)
    order = np.argsort(max_corrs)[::-1]
    return order, max_corrs

def plot_corr_summary(correlations: np.ndarray, byte_pos: int, outdir: str):
    max_corrs = np.abs(correlations).max(axis=1)
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

# ---------- Utilities ----------

def inv_shiftrows_permutation(key16: List[int]) -> List[int]:
    """
    Permute a 16-byte key that was recovered in ciphertext byte order (post-ShiftRows)
    back into the 'state' (column-major) order expected by the key schedule.
    Use this if your reconstructed K0 looks wrong after inv_key_schedule_from_k10.
    """
    # build mapping by simulating AES ShiftRows: state is 4x4 column-major:
    # index = 4*c + r  (r=0..3 rows, c=0..3 cols)
    # after ShiftRows, row r is rotated left by r
    permuted = [0] * 16
    for r in range(4):
        for c in range(4):
            before_idx = 4*c + r
            after_c = (c + r) % 4
            after_idx = 4*after_c + r
            # if we recovered bytes in after_idx order, then to get "before" we take recovered[after_idx]
            permuted[before_idx] = key16[after_idx]
    return permuted

# ---------- Orchestration ----------

def solve_problem1_aligned(csv_filename: str = "real_power_trace.csv",
                           npy_filename: str = "hw_aligned_traces.npy",
                           outdir_root: str = "results_problem1_aligned",
                           use_inv_sbox: bool = True,
                           try_shiftrows_fix: bool = True):
    print("\n=== SOLVING PROBLEM 1 (Aligned Traces): Last-Round CPA ===")
    ciphertexts, traces = load_ciphertexts_and_traces(csv_filename, npy_filename)

    results_dir = outdir_root
    os.makedirs(results_dir, exist_ok=True)

    recovered_k10: List[int] = []
    per_byte_correlations = []

    for byte_pos in range(16):
        print(f"\nAttacking key byte {byte_pos} ...")
        best_key, best_corr, correlations = cpa_last_round(traces, ciphertexts, byte_pos, use_inv_sbox=use_inv_sbox)
        recovered_k10.append(best_key)
        per_byte_correlations.append(correlations)

        # Rank & save in requested format
        ranks, max_corrs = rank_candidates(correlations)
        print("  Top 5 candidates:")
        for r, k in enumerate(ranks[:5], start=1):
            print(f"    Rank {r:2d}: 0x{k:02X} (corr={max_corrs[k]:.6f})")

        with open(os.path.join(results_dir, f"byte_{byte_pos:02d}_rank.txt"), "w") as f:
            for r, k in enumerate(ranks, start=1):
                f.write(f"Rank {r:3d}: 0x{k:02X} (corr={max_corrs[k]:.6f})\n")

        # Plot per-byte correlation summary
        plot_corr_summary(correlations, byte_pos, results_dir)

    # Compose K10
    k10_hex = "".join(f"{b:02X}" for b in recovered_k10)
    print(f"\nRecovered Round-10 Key (K10) (raw order): {k10_hex}")

    # Optionally try ShiftRows fix (permute bytes) if reconstruction looks wrong
    k10_for_schedule = recovered_k10
    if try_shiftrows_fix:
        print("\nAttempting inv-ShiftRows permutation (if needed) to match key-schedule order...")
        permuted = inv_shiftrows_permutation(recovered_k10)
        perm_hex = "".join(f"{b:02X}" for b in permuted)
        print(f"Permuted K10 (candidate): {perm_hex}")
        # Heuristic: let user decide which to use; here we try reconstructing from permuted first,
        # and if that looks invalid we fallback to raw order.
        round_keys_permuted = inv_key_schedule_from_k10(permuted)
        k0_p_hex = "".join(f"{b:02X}" for b in round_keys_permuted[0])
        print(f"Reconstructed K0 from permuted K10: {k0_p_hex}")
        # Also reconstruct from raw order for comparison
        round_keys_raw = inv_key_schedule_from_k10(recovered_k10)
        k0_r_hex = "".join(f"{b:02X}" for b in round_keys_raw[0])
        print(f"Reconstructed K0 from raw K10     : {k0_r_hex}")

        # choose permuted as default if its K0 differs (you can change logic)
        # We will keep both results in files; user can decide which is correct (or verify using ciphertext)
        chosen_round_keys = round_keys_permuted
    else:
        chosen_round_keys = inv_key_schedule_from_k10(k10_for_schedule)

    # Save keys summary + per-round keys (both permutations if computed)
    with open(os.path.join(results_dir, "recovered_keys_summary.txt"), "w") as f:
        f.write(f"K10 (raw order) : {''.join(f'{b:02X}' for b in recovered_k10)}\n")
        if try_shiftrows_fix:
            f.write(f"K10 (permuted)  : {''.join(f'{b:02X}' for b in permuted)}\n")
        f.write("\n")
        for r, rk in enumerate(chosen_round_keys):
            f.write(f"Round {r:02d} Key: {''.join(f'{b:02X}' for b in rk)}\n")

    # Also save each round key separately if you want strict deliverables
    for r, rk in enumerate(chosen_round_keys):
        with open(os.path.join(results_dir, f"round_{r:02d}_key.txt"), "w") as f:
            f.write("".join(f"{b:02X}" for b in rk) + "\n")

    print(f"\nAll results saved in: {results_dir}/")

    # Optional verification if pycryptodome is installed and CSV includes plaintexts
    try:
        from Crypto.Cipher import AES  # type: ignore
        try:
            # attempt to verify first row ciphertext matches encryption of plaintext using recovered K0
            df = pd.read_csv(csv_filename, header=None)
            pt_hex = str(df.iloc[0, 0]).strip()
            ct_hex = str(df.iloc[0, 1]).strip()
            pt0 = bytes.fromhex(pt_hex)
            ct0 = bytes.fromhex(ct_hex)

            # try both candidate K0s if available
            candidates = [chosen_round_keys[0]]
            if try_shiftrows_fix and round_keys_raw[0] != chosen_round_keys[0]:
                candidates.append(round_keys_raw[0])

            verified = False
            for cand in candidates:
                key = bytes(cand)
                cipher = AES.new(key, AES.MODE_ECB)
                enc = cipher.encrypt(pt0)
                if enc == ct0:
                    print("SUCCESS: Verified K0 by encryption of first plaintext -> ciphertext.")
                    print("Verified Master Key (K0):", ''.join(f"{b:02X}" for b in cand))
                    verified = True
                    break
            if not verified:
                print("WARNING: None of the K0 candidates produced the first ciphertext from the first plaintext.")
                print("This could mean: wrong permutation, wrong leakage model, insufficient traces, or CSV ordering mismatch.")
        except Exception as e:
            print("Verification attempted but failed (couldn't parse plaintext/ciphertext pair):", e)
    except Exception:
        print("PyCryptodome not installed; skipping automatic K0 verification. Install with `pip install pycryptodome` to enable.")

    return recovered_k10, chosen_round_keys, per_byte_correlations

if __name__ == "__main__":
    # You can toggle use_inv_sbox=False to try SBOX-based leakage hypothesis
    recovered_k10, round_keys, per_byte_correlations = solve_problem1_aligned(
        csv_filename="real_power_trace.csv",
        npy_filename="hw_aligned_traces.npy",
        outdir_root="results_problem1_aligned",
        use_inv_sbox=True,
        try_shiftrows_fix=True
    )
    print("\nDone.")
