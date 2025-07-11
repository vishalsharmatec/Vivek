import pandas as pd

df = pd.read_parquet(r"data\User_detail_report.parquet")
print(df.head())