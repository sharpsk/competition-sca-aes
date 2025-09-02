import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import correlate

# ------------------------------
# Load traces
# ------------------------------
df = pd.read_csv("real_power_trace.csv")

# Drop non-numeric columns (first 2 are hex + unnamed cols)
trace_data = df.drop(columns=df.columns[:2])
trace_data = trace_data.drop(columns=[c for c in trace_data.columns if "Unnamed" in c])
traces = trace_data.to_numpy(dtype=float)

print("Traces shape:", traces.shape)

# Reference trace
ref_trace = traces[0]

def align_trace(trace, ref_trace):
    corr = correlate(trace, ref_trace, mode='full')
    lag = np.argmax(corr) - (len(ref_trace) - 1)
    if lag > 0:
        aligned = np.pad(trace, (lag, 0), mode='constant')[:len(trace)]
    elif lag < 0:
        aligned = np.pad(trace, (0, -lag), mode='constant')[:len(trace)]
    else:
        aligned = trace
    return aligned

# Align all traces
aligned_traces = np.array([align_trace(t, ref_trace) for t in traces])
print("Aligned traces shape:", aligned_traces.shape)

# ------------------------------
# Compute mean traces
# ------------------------------
mean_before = np.mean(traces, axis=0)
mean_after = np.mean(aligned_traces, axis=0)

# ------------------------------
# Plot overlay
# ------------------------------
plt.figure(figsize=(12, 6))
plt.plot(mean_before, label="Mean Before Alignment", alpha=0.7)
plt.plot(mean_after, label="Mean After Alignment", alpha=0.7)
plt.title("Mean Trace Overlay (Before vs After Alignment)")
plt.xlabel("Sample Points")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)
plt.show()
