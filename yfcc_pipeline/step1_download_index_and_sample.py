"""
Step 1 of the YFCC-DINO pipeline

Streams the mehdidc/yfcc15m dataset from HuggingFace, which contains
the full YFCC metadata including real upload timestamps and download
URLs. Samples N_PER_YEAR images per year for 2007-2013.

Output: data/yfcc_sampled/sampled_urls.csv
    columns: photoid, year, month, downloadurl, dateuploaded

Requirements:
    pip install datasets pandas tqdm
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm

# parameters
OUTPUT_DIR   = 'data/yfcc_sampled'
N_PER_YEAR   = 20000    # images per year
YEARS        = list(range(2007, 2014))
RANDOM_SEED  = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)
np.random.seed(RANDOM_SEED)

# stream dataset
print("Loading mehdidc/yfcc15m from HuggingFace (streaming)...")

from datasets import load_dataset

ds = load_dataset("mehdidc/yfcc15m", split="train", streaming=True)

# reservoir sampling per year: keeps N_PER_YEAR samples without
# loading all rows into memory
reservoirs = {y: [] for y in YEARS}
counts     = {y: 0  for y in YEARS}
total_seen = 0

import random
random.seed(RANDOM_SEED)

for item in tqdm(ds, desc="Streaming", unit=" rows"):
    total_seen += 1

    # get upload timestamp
    ts = item.get('dateuploaded')
    if not ts:
        continue

    try:
        year  = datetime.fromtimestamp(int(ts)).year
        month = datetime.fromtimestamp(int(ts)).month
    except (ValueError, OSError, OverflowError):
        continue

    if year not in YEARS:
        continue

    url = item.get('downloadurl', '')
    if not url or url == 'None':
        continue

    photo_id = item.get('photoid')

    # reservoir sampling (Algorithm R)
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

    # stop early if all years are full
    if all(len(reservoirs[y]) >= N_PER_YEAR for y in YEARS):
        print(f"\n  All years full after {total_seen:,} rows — stopping early")
        break

# save
rows = []
for year, reservoir in reservoirs.items():
    rows.extend(reservoir)

df = pd.DataFrame(rows)
out_path = os.path.join(OUTPUT_DIR, 'sampled_urls.csv')
df.to_csv(out_path, index=False)

print(f"\nSaved {len(df):,} URLs to {out_path}")
print("\nYear distribution:")
print(df.groupby('year').size().to_string())
print("\nMonth distribution (sanity check for seasonal content):")
print(df.groupby(['year', 'month']).size().unstack(fill_value=0).to_string())
print("\nStep 1 complete.")