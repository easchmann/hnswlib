"""
Merges the per-task manifest CSVs written by step2 into a single
downloaded_manifest.csv.

Usage:
    python yfcc_pipeline/merge_manifests.py
"""

import os
import glob
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--manifest_dir', default='data/yfcc_sampled')
parser.add_argument('--output',       default='data/yfcc_sampled/downloaded_manifest.csv')
args = parser.parse_args()

pattern = os.path.join(args.manifest_dir, 'downloaded_manifest_task*.csv')
files   = sorted(glob.glob(pattern))

if not files:
    print(f"no partial manifests found at {pattern}")
    raise SystemExit(1)

print(f"merging {len(files)} partial manifests...")
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
df = df.drop_duplicates(subset='photo_id')
df.to_csv(args.output, index=False)

print(f"saved {len(df):,} rows to {args.output}")
print(df.groupby('year').size().to_string())
