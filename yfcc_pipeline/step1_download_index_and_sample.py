"""
Streams the mehdidc/yfcc15m dataset from HuggingFace and reservoir-samples
N_PER_YEAR images per year for 2007-2013. Targeting ~5M images total.

Output: data/yfcc_sampled/sampled_urls.csv
    columns: photoid, year, month, downloadurl, dateuploaded

"""

import os
import random
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm
from datasets import load_dataset

OUTPUT_DIR  = 'data/yfcc_sampled'
N_PER_YEAR  = 750_000   # 750k x 7 years = ~5.25M total; fits comfortably in ~800GB scratch
YEARS       = list(range(2007, 2014))
RANDOM_SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

print(f"sampling {N_PER_YEAR:,}/year x {len(YEARS)} years = ~{N_PER_YEAR*len(YEARS)/1e6:.1f}M images")

ds = load_dataset("mehdidc/yfcc15m", split="train", streaming=True)

# reservoir sampling per year (Algorithm R) - keeps sample unbiased
# without loading everything into memory
reservoirs = {y: [] for y in YEARS}
counts     = {y: 0  for y in YEARS}
total_seen = 0

for item in tqdm(ds, desc="streaming", unit=" rows"):
    total_seen += 1

    ts = item.get('dateuploaded')
    if not ts:
        continue
    try:
        dt    = datetime.fromtimestamp(int(ts))
        year  = dt.year
        month = dt.month
    except (ValueError, OSError, OverflowError):
        continue

    if year not in YEARS:
        continue

    url = item.get('downloadurl', '')
    if not url or url == 'None':
        continue

    photo_id = item.get('photoid')
    counts[year] += 1

    if len(reservoirs[year]) < N_PER_YEAR:
        reservoirs[year].append({
            'photoid':      photo_id,
            'year':         year,
            'month':        month,
            'downloadurl':  url,
            'dateuploaded': ts,
        })
    else:
        j = random.randint(0, counts[year] - 1)
        if j < N_PER_YEAR:
            reservoirs[year][j] = {
                'photoid':      photo_id,
                'year':         year,
                'month':        month,
                'downloadurl':  url,
                'dateuploaded': ts,
            }

    if total_seen % 1_000_000 == 0:
        filled = {y: len(reservoirs[y]) for y in YEARS}
        print(f"  {total_seen/1e6:.0f}M rows seen, reservoirs: {filled}")

print(f"\ndone streaming, saw {total_seen:,} rows total")

rows = []
for year, reservoir in reservoirs.items():
    rows.extend(reservoir)

df = pd.DataFrame(rows)
out_path = os.path.join(OUTPUT_DIR, 'sampled_urls.csv')
df.to_csv(out_path, index=False)

print(f"\nsaved {len(df):,} rows to {out_path}")
print(df.groupby('year').size().to_string())
print("\nmonth distribution (sanity check):")
print(df.groupby(['year', 'month']).size().unstack(fill_value=0).to_string())
