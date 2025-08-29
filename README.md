# competition-sca-aes

# IIT Kharagpur SCA Competition 2025 – Problem 1 (Simulated AES Traces)

This repository contains our solution for **Problem 1** of the IIT Kharagpur Side-Channel Analysis (SCA) Competition 2025.  
The task is to recover the secret AES key from **simulated power traces of an unmasked AES implementation**.

## 📂 Contents
- **Code**
  - Implementation of **Correlation Power Analysis (CPA)** attack using ciphertexts.
  - Scripts to compute key-byte rankings based on correlation scores.
  - Automated generation of sixteen output files (`byte_00.txt … byte_15.txt`) containing ranked key hypotheses.

- **Deliverables**
  - Final guessed 16-byte AES key.
  - Ranked candidate lists for each key byte.

## ⚡ Methods Used
- **CPA (Correlation Power Analysis)** applied directly using ciphertext values.
- Hamming Weight (HW) model used for leakage hypothesis.
- No preprocessing applied — traces were used as provided.
- Attack targeted the last round of AES using ciphertexts.

## 📖 Competition Reference
IIT Kharagpur SCA Competition 2025 (Online Phase: till Aug 30).
