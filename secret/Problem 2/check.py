import pandas as pd

df = pd.read_csv("real_power_trace.csv")
print(df.head())       # show first 5 rows
print(df.columns)      # show all column names
print(df.shape)        # show (rows, columns)
