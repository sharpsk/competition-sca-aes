# Bit by Bit Side Channel Hackathon 2025  
## AES Key Recovery via Side Channel Analysis

This repository contains our submission for the **Bit by Bit Side Channel Hackathon 2025**.  
We tackled two problems involving side-channel analysis of AES encryption.  

Our approaches were based primarily on **Correlation Power Analysis (CPA)** with minimal preprocessing (Problem 1) and extended preprocessing pipelines (Problem 2).  

---

## Repository Structure
secret/
│── problem1/ → Solution for Problem 1 (simulated traces, successful key recovery)
│── problem2/ → Solution for Problem 2 (real traces, preprocessing + CPA, partial results)
│── README.md → Main repository README (this file)

---

## Problems Overview

### 🔹 Problem 1 — Simulated AES Power Traces
- Dataset: 5,000 traces (ciphertext + simulated power values)  
- Approach: CPA with Hamming Weight leakage model  
- Result: **✅ Successfully recovered AES-128 key**  
- See [`problem1/README.md`](./problem1/README.md) for details  

---

### 🔹 Problem 2 — Real AES Power Traces
- Dataset: Real measured traces (plaintext + ciphertext + power samples)  
- Approach: Preprocessing (alignment, normalization, denoising) + CPA  
- Result: **❌ Key recovery unsuccessful** (candidate keys derived but mismatched)  
- See [`problem2/README.md`](./problem2/README.md) for details  

---

## Team
- **Surya K**  
- **Renita J**  
- Advisor: Ms. Suganya A  

---

## Notes
- Attack scripts are provided in both problem folders.  
- All ranked key byte files are included (`byte_00.txt` → `byte_15.txt`).  
- Reports are documented in detail and reproducible with the provided code.  

---
