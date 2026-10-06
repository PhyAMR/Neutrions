
import pandas as pd
import pyarrow.parquet as pq

events = pq.read_table('data/raw/training/events.parquet').to_pandas()
truth = pq.read_table('data/raw/training/truth.parquet').to_pandas()
print("Events DataFrame shape:", events.shape)
print("Truth DataFrame shape:", truth.shape)
print("Events DataFrame columns:", events.columns)
print("Truth DataFrame columns:", truth.columns)

print("Events DataFrame:")
print(events.head())
print("Truth DataFrame:")
print(truth.head())

print("Truth DataFrame unique labels:", truth['label_name'].unique())
print("Truth DataFrame unique pairs:", truth['pair_id'].unique())
print("Truth DataFrame unique labels id:", truth['label'].unique())
