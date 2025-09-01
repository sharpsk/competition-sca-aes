# Problem 1 — Simulated AES Power Traces

## Overview
This folder contains our solution for **Problem 1** of the *Bit by Bit Side Channel Hackathon 2025*.  
We were provided with **5,000 simulated side-channel power traces** of an unmasked AES implementation. Each trace contained ciphertext and corresponding simulated power consumption values.  

Our objective was to recover the 128-bit AES secret key using side-channel analysis.  

---

## Methodology
- **Attack Type**: Correlation Power Analysis (CPA)  
- **Target Round**: Last round of AES (10th round)  
- **Leakage Model**: Hamming Weight of S-box output  
- **Correlation Metric**: Pearson correlation coefficient  

Steps:
1. Parse ciphertext + trace data from CSV.  
2. Generate hypothetical leakage for all 256 key guesses per byte.  
3. Compute correlation between hypothetical leakage and trace samples.  
4. Identify the key byte with the maximum correlation.  
5. Rank all 256 key candidates for each byte.  

---


