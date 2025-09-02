import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load original traces (CSV) and aligned traces (NPY)
orig_traces = pd.read_csv("real_power_trace.csv", header=None).values
aligned_traces = np.load("hw_aligned_traces.npy")

print("Original traces shape:", orig_traces.shape)
print("Aligned traces shape:", aligned_traces.shape)

# Select a few traces for visualization
idxs = [0, 10, 50, 100]  # you can change indices
plt.figure(figsize=(14, 6))

# Before alignment
plt.subplot(1, 2, 1)
for idx in idxs:
    plt.plot(orig_traces[idx][:2000], alpha=0.7)  # show first 2000 points
plt.title("Before Alignment")
plt.xlabel("Sample index")
plt.ylabel("Amplitude")

# After alignment
plt.subplot(1, 2, 2)
for idx in idxs:
    plt.plot(aligned_traces[idx], alpha=0.7)
plt.title("After Alignment")
plt.xlabel("Sample index")
plt.ylabel("Amplitude")

plt.tight_layout()
plt.show()
