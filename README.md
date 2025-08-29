# competition-sca-aes

# IIT Kharagpur SCA Competition 2025 – Problem 1 (Simulated AES Traces)

This repository contains our solution for **Problem 1** of the IIT Kharagpur Side-Channel Analysis (SCA) Competition 2025.  
The task is to recover the secret AES key from **simulated power traces of an unmasked AES implementation**.

## 📂 Contents
- **Code**
  - Preprocessing scripts for simulated traces.
  - Deep learning models (MLP/CNN) for profiling attacks.
  - Key recovery and Guessing Entropy (GE) evaluation.
  - Correlation Power Analysis (CPA) as a baseline.

- **Deliverables**
  - Final guessed AES key.
  - Sixteen ranked candidate lists (`byte_00.txt … byte_15.txt`).

- **Documentation**
  - Report describing preprocessing, training process, model architecture, and evaluation.

## ⚡ Methods Used
- **Deep Learning (MLP/CNN)** to classify AES intermediate values under Hamming Weight leakage model.
- **Guessing Entropy (GE)** as the evaluation metric.
- Trace normalization and preprocessing for stable training.
- Comparison with classical **CPA attack**.

## 📖 Competition Reference
IIT Kharagpur SCA Competition 2025 (Online Phase: till Aug 30).  
