import pandas as pd

file_path = "real_power_trace.csv"

# Read as tab-separated values
df = pd.read_csv(file_path, sep="\t", header=None)

# Extract plaintexts, ciphertexts, and traces
plaintexts = df.iloc[:, 0].tolist()
ciphertexts = df.iloc[:, 1].tolist()
traces = df.iloc[:, 2:].astype(float).values  # convert to numpy array

print("Plaintexts:", plaintexts[0])
print("Ciphertexts:", ciphertexts[0])
print("Trace shape:", traces.shape)
print("First trace sample:", traces[0][:20])
