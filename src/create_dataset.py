import pandas as pd

import json


df = pd.read_csv('../data/TMDb_updated.csv')

df = df.dropna()

df_data = df[['title', 'overview']]

data = []

for _, row in df_data.iterrows():

    text = f"[DESC] {row['overview']} [TITLE] {row['title']} [END]"
    data.append(text)

with open('../data/data.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, indent = 2)

print(f"Saved {len(data)} rows")