#!/usr/bin/env python3
# hw_ml_kr.py
# Correlation Power Analysis + ML-based Key Recovery

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")        # prevent Tkinter GUI crashes
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# AES S-box
SBOX = [
    # 0     1      2      3      4      5      6      7      8      9      A      B      C      D      E      F
    0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,  # 0
    0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,  # 1
    0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,  # 2
    0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,  # 3
    0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,  # 4
    0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,  # 5
    0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,  # 6
    0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,  # 7
    0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,  # 8
    0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,  # 9
    0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,  # A
    0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,  # B
    0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,  # C
    0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,  # D
    0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,  # E
    0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16   # F
]

# Hamming weight lookup
HW = [bin(x).count("1") for x in range(256)]

# ---------------- Data Loader ----------------
def load_data():
    df = pd.read_csv("real_power_trace.csv", sep="\t", header=None)

    # Column 0 = plaintext, Column 1 = ciphertext
    plaintexts = df.iloc[:,0].apply(lambda x: bytes.fromhex(x.strip())).tolist()
    ctexts     = df.iloc[:,1].apply(lambda x: bytes.fromhex(x.strip())).tolist()

    plaintexts = np.array([list(b) for b in plaintexts], dtype=np.uint8)
    ctexts     = np.array([list(b) for b in ctexts], dtype=np.uint8)

    # Columns 2..end = traces
    traces = df.iloc[:,2:].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    traces = np.nan_to_num(traces, nan=0.0, posinf=0.0, neginf=0.0)

    return plaintexts, ctexts, traces

# ---------------- CPA Attack ----------------
def cpa_attack(plaintexts, traces):
    n_traces, n_samples = traces.shape
    key_guesses = np.zeros(16, dtype=np.uint8)

    print(f"Loaded {n_traces} traces with {n_samples} samples each")

    for byte_idx in range(16):
        max_corr = -1
        best_guess = 0
        corrs = []

        for kguess in range(256):
            hyp = np.array([HW[SBOX[p[byte_idx] ^ kguess]] for p in plaintexts], dtype=np.float32)
            hyp -= np.mean(hyp)

            corr = np.max(np.abs(np.corrcoef(hyp, traces, rowvar=False)[0,1:]))
            corrs.append((kguess, corr))

            if corr > max_corr:
                max_corr = corr
                best_guess = kguess

        corrs.sort(key=lambda x: x[1], reverse=True)
        print(f"\nKey byte {byte_idx}: CPA Top 5")
        for rank,(kg,c) in enumerate(corrs[:5], start=1):
            print(f"  Rank {rank}: 0x{kg:02X} (corr={c:.4f})")

        key_guesses[byte_idx] = best_guess

    return key_guesses

# ---------------- ML Attack ----------------
def ml_attack(plaintexts, traces):
    n_traces, _ = traces.shape
    recovered_key = np.zeros(16, dtype=np.uint8)

    for byte_idx in range(16):
        labels = np.array([SBOX[p[byte_idx] ^ k] for p,k in zip(plaintexts, [0]*n_traces)], dtype=np.uint8)
        labels = np.array([HW[val] for val in labels], dtype=np.uint8)

        X_train,X_test,y_train,y_test = train_test_split(traces, labels, test_size=0.2, random_state=42)

        clf = LogisticRegression(max_iter=200, solver="lbfgs", multi_class="auto")
        clf.fit(X_train,y_train)
        preds = clf.predict(X_test)
        acc = accuracy_score(y_test,preds)

        print(f"[ML] Byte {byte_idx}: classifier accuracy = {acc:.3f}")

        # Simple heuristic: pick HW with max mean prediction
        hw_pred = np.bincount(preds).argmax()
        candidate_keys = [k for k in range(256) if HW[SBOX[0 ^ k]] == hw_pred]
        if candidate_keys:
            recovered_key[byte_idx] = candidate_keys[0]

    return recovered_key

# ---------------- Main ----------------
def run_attack():
    print("\n=== SOLVING PROBLEM 1 (Aligned Traces): CPA + ML-based Key Recovery ===")
    plaintexts, ctexts, traces = load_data()

    # CPA
    cpa_key = cpa_attack(plaintexts, traces)
    print("\nRecovered Key (CPA):", "".join(f"{b:02X}" for b in cpa_key))

    # ML
    ml_key = ml_attack(plaintexts, traces)
    print("\nRecovered Key (ML):", "".join(f"{b:02X}" for b in ml_key))

if __name__ == "__main__":
    run_attack()
