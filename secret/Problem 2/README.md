# Problem 2 — Real AES Power Traces

## Overview
This folder contains our solution for **Problem 2** of the *Bit by Bit Side Channel Hackathon 2025*.  
We were provided with **real hardware power traces** of an unmasked AES implementation. Each trace contained plaintext, ciphertext, and corresponding real measured power samples.  

Our objective was to recover the AES-128 secret key under realistic, noisy conditions.  

**Status: ❌ Unsuccessful Key Recovery (attempted)**

---

## Preprocessing
The real traces contained misalignments and noise, so the following preprocessing pipeline was implemented:

1. **Bad-trace detection** using z-score thresholds.  
2. **Baseline subtraction** (first 100 samples).  
3. **Trace alignment** (cross-correlation, ±50 sample shift).  
4. **Column-wise normalization**.  
5. **Optional smoothing** (Savitzky–Golay filter).  
6. **Optional denoising** (SVD, retaining 90% energy).  

---

## Attack Methodology
- **Attack Type**: CPA (Correlation Power Analysis)  
- **Target Round**: 10th round AES (last round)  
- **Leakage Model**: Hamming weight of inverse S-box output  
- **Correlation Metric**: Pearson correlation per sample  
- **Points of Interest (POIs)**: Selected based on maximum correlation  
- **Refinement**: Candidate re-scoring with expanded POI  

---

## Results
Two attempts were made:  

- **Without SVD denoising**  
  - Recovered K10: `99989E871EADCBD1A51431BB298D1D0C`  
  - Reconstructed AES-128 K0: `824E888CF0AD1CC6FF0C850FD37D27F9`  

- **With SVD denoising**  
  - Recovered K10: `6F989E871EADCBD1A514B0CC298D1D0C`  
  - Reconstructed AES-128 K0: `78DBA2144F4C5002E43FFFC2A21BDC43`  

Neither matched the true secret key.  

---
