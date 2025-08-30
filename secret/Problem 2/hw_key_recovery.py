#!/usr/bin/env python3
"""
AES Side Channel Analysis Challenge Solver - Problem 1 (Aligned Traces)
Loads ciphertexts from CSV and aligned traces from NPY.
Performs last round CPA, saves ranked candidates, and reconstructs AES round keys.
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from typing import Tuple, List
from src.aes_utils import AESHelper


class AESProblem1Solver:
    """Solver for AES side channel analysis challenge - Problem 1 (Aligned traces)."""

    def __init__(self):
        self.aes_helper = AESHelper()

    # ------------- IO -------------

    def load_aligned_traces(self, csv_filename: str, npy_filename: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load aligned traces (from .npy) and ciphertexts (from .csv).
        CSV format: plaintext(hex), ciphertext(hex), trace samples...
        Returns:
          traces (float32)     shape = (num_traces, num_samples)
          ciphertexts (uint8)  shape = (num_traces, 16)
        """
        print(f"Loading ciphertexts from {csv_filename}...")
        df = pd.read_csv(csv_filename, header=None)

        ciphertexts = df.iloc[:, 1].apply(lambda x: bytes.fromhex(x)).values
        ciphertexts = np.array([list(ct) for ct in ciphertexts], dtype=np.uint8)

        print(f"Loading aligned traces from {npy_filename}...")
        traces = np.load(npy_filename).astype(np.float32)

        # Ensure matching number of traces
        num = min(len(traces), len(ciphertexts))
        traces = traces[:num]
        ciphertexts = ciphertexts[:num]

        print(f"Loaded {len(traces)} aligned traces with {traces.shape[1]} samples each")
        return traces, ciphertexts

    # ------------- CPA (last round) -------------

    def cpa_attack_last_round(self, traces: np.ndarray, ciphertexts: np.ndarray,
                              byte_pos: int) -> Tuple[int, float, np.ndarray]:
        """CPA attack on last round (inverse S-box on ciphertext). Returns best key byte."""
        num_traces, num_samples = traces.shape
        correlations = np.zeros((256, num_samples), dtype=np.float32)

        ct_byte = ciphertexts[:, byte_pos]
        for key_guess in range(256):
            # Last round: invSbox(C_i ^ k10_i)
            inter = self.aes_helper.inv_sbox_lookup((ct_byte ^ key_guess).astype(np.uint8))
            # Hamming-weight leakage model
            hw_model = np.unpackbits(inter.reshape(-1, 1), axis=1).sum(axis=1).astype(np.float32)

            # Pearson correlation (abs) per sample
            x = hw_model
            x_mean = x.mean()
            x_std = x.std()
            if x_std == 0:
                continue
            y = traces
            y_mean = y.mean(axis=0)
            y_std = y.std(axis=0)
            denom = x_std * y_std
            valid = denom != 0
            corr = np.zeros(num_samples, dtype=np.float32)
            corr[valid] = np.abs(((y[:, valid] - y_mean[valid]) * (x[:, None] - x_mean)).sum(axis=0) /
                                 ((len(x) - 1) * denom[valid]))
            correlations[key_guess, :] = corr

        max_corr_per_guess = correlations.max(axis=1)
        best_key = int(np.argmax(max_corr_per_guess))
        max_correlation = float(max_corr_per_guess[best_key])
        return best_key, max_correlation, correlations

    def rank_key_candidates(self, correlations: np.ndarray) -> np.ndarray:
        """Rank key guesses by max correlation (highest first)."""
        max_corrs = np.max(correlations, axis=1)
        return np.argsort(max_corrs)[::-1], max_corrs

    def plot_correlations(self, correlations: np.ndarray, byte_pos: int, save_dir: str):
        """Plot correlation vs key guess for each byte."""
        max_corrs = np.max(correlations, axis=1)
        plt.figure(figsize=(10, 5))
        plt.plot(range(256), max_corrs, marker="o", markersize=2, linewidth=1)
        plt.title(f"CPA Correlation for Byte {byte_pos}")
        plt.xlabel("Key Guess (0–255)")
        plt.ylabel("Max Correlation")
        plt.grid(True)

        best_key = int(np.argmax(max_corrs))
        plt.axvline(best_key, color="r", linestyle="--", label=f"Best Guess: 0x{best_key:02X}")
        plt.legend()

        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(f"{save_dir}/byte_{byte_pos:02d}_correlation.png", dpi=150)
        plt.close()

    # ------------- AES-128 inverse key schedule -------------

    @staticmethod
    def _split_words(key16: List[int]) -> List[List[int]]:
        return [key16[i:i+4] for i in range(0, 16, 4)]

    @staticmethod
    def _merge_words(words: List[List[int]]) -> List[int]:
        return [b for w in words for b in w]

    @staticmethod
    def _rot_word(w: List[int]) -> List[int]:
        return [w[1], w[2], w[3], w[0]]  # left rotate

    def _sub_word(self, w: List[int]) -> List[int]:
        arr = np.array(w, dtype=np.uint8)
        sub = self.aes_helper.sbox_lookup(arr).astype(np.uint8)
        return list(map(int, sub.tolist()))

    def reconstruct_round_keys_from_k10(self, k10: List[int]) -> List[List[int]]:
        """Reconstruct K0..K10 from round-10 key."""
        RCON = [0x00, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1B, 0x36]
        W: List[List[int]] = [None] * 44
        words10 = self._split_words(k10)
        for j in range(4):
            W[40 + j] = words10[j]

        for i in range(43, 3, -1):
            if i % 4 == 0:
                temp = self._sub_word(self._rot_word(W[i - 1]))
                temp[0] ^= RCON[i // 4]
                W[i - 4] = [W[i][k] ^ temp[k] for k in range(4)]
            else:
                W[i - 4] = [W[i][k] ^ W[i - 1][k] for k in range(4)]

        round_keys: List[List[int]] = []
        for r in range(11):
            rk = self._merge_words([W[4 * r + j] for j in range(4)])
            round_keys.append(rk)
        return round_keys

    # ------------- Orchestration -------------

    def solve_problem1_aligned(self, csv_filename: str, npy_filename: str) -> Tuple[List[int], List[List[int]]]:
        print("\n=== SOLVING PROBLEM 1: Aligned Power Traces ===")
        traces, ciphertexts = self.load_aligned_traces(csv_filename, npy_filename)

        results_dir = "results_problem1_aligned"
        os.makedirs(results_dir, exist_ok=True)

        recovered_k10: List[int] = []

        for byte_pos in range(16):
            print(f"\nAttacking key byte {byte_pos}...")
            best_key, max_corr, correlations = self.cpa_attack_last_round(traces, ciphertexts, byte_pos)
            recovered_k10.append(best_key)

            # Rank candidates
            ranks, max_corrs = self.rank_key_candidates(correlations)

            # Print top 5 ranked keys
            print("  Top 5 candidates (Rank, Key, Correlation):")
            for rank, key_guess in enumerate(ranks[:5], start=1):
                print(f"    Rank {rank}: 0x{key_guess:02X} (corr={max_corrs[key_guess]:.4f})")

            # Save ranked candidates file
            with open(os.path.join(results_dir, f"byte_{byte_pos:02d}_rank.txt"), "w") as f:
                for rank, key_guess in enumerate(ranks, start=1):
                    f.write(f"Rank {rank:3d}: 0x{key_guess:02X} (corr={max_corrs[key_guess]:.6f})\n")

            # Plot correlation distribution
            self.plot_correlations(correlations, byte_pos, results_dir)

        # Recovered Round 10 key
        k10_hex = "".join(f"{b:02X}" for b in recovered_k10)
        print(f"\nRecovered 10th Round Key (K10): {k10_hex}")

        # Reconstruct Master Key
        round_keys = self.reconstruct_round_keys_from_k10(recovered_k10)
        k0_hex = "".join(f"{b:02X}" for b in round_keys[0])
        print(f"Reconstructed AES Master Key (K0): {k0_hex}")

        # Save all round keys
        with open(os.path.join(results_dir, "round_keys.txt"), "w") as f:
            f.write(f"K10 (Round 10): {k10_hex}\n")
            f.write(f"K0  (Master) : {k0_hex}\n\n")
            for r, rk in enumerate(round_keys):
                rk_hex = "".join(f"{b:02X}" for b in rk)
                f.write(f"Round {r:02d} Key: {rk_hex}\n")

        print(f"All results saved in {results_dir}/")
        return recovered_k10, round_keys


def main():
    solver = AESProblem1Solver()
    solver.solve_problem1_aligned("real_power_trace.csv", "hw_aligned_traces.npy")


if __name__ == "__main__":
    main()
