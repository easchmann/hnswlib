"""
Downloads images from the Flickr URLs sampled in step 1.

Processes the CSV in chunks and flushes a partial manifest after each one,
so it's safe to kill and resume at any point. Also supports Slurm job arrays
via --task_id / --num_tasks so the work can be split across multiple nodes.

run merge_manifests.py to consolidate after all array tasks finish

Output:
    data/yfcc_sampled/images/{year}/{photo_id}.jpg
    data/yfcc_sampled/downloaded_manifest.csv  (or per-task partial CSVs)

Requirements:
    pip install requests tqdm pandas pillow
"""

import os
import time
import argparse
import requests
import pandas as pd
from io import BytesIO
from tqdm import tqdm
from PIL import Image, ImageFile
from concurrent.futures import ThreadPoolExecutor, as_completed

ImageFile.LOAD_TRUNCATED_IMAGES = True

parser = argparse.ArgumentParser()
parser.add_argument('--sampled_csv',  default='data/yfcc_sampled/sampled_urls.csv')
parser.add_argument('--image_dir',    default='data/yfcc_sampled/images')
parser.add_argument('--manifest_out', default='data/yfcc_sampled/downloaded_manifest.csv')
parser.add_argument('--n_workers',    type=int, default=64)
parser.add_argument('--timeout',      type=int, default=15)
parser.add_argument('--min_kb',       type=int, default=5)
parser.add_argument('--chunk_size',   type=int, default=50_000)
parser.add_argument('--task_id',      type=int, default=0)
parser.add_argument('--num_tasks',    type=int, default=1)
args = parser.parse_args()

os.makedirs(args.image_dir, exist_ok=True)

df_all = pd.read_csv(args.sampled_csv)
print(f"loaded {len(df_all):,} URLs")

# split work across array tasks if running as a Slurm array
if args.num_tasks > 1:
    df_all = df_all.iloc[args.task_id::args.num_tasks].reset_index(drop=True)
    print(f"task {args.task_id}/{args.num_tasks}: {len(df_all):,} rows")
    task_manifest = args.manifest_out.replace('.csv', f'_task{args.task_id:04d}.csv')
else:
    task_manifest = args.manifest_out

print(df_all.groupby('year').size().to_string())


def download_one(row):
    year     = int(row['year'])
    month    = int(row.get('month', 0))
    photo_id = str(row['photoid'])
    url      = str(row['downloadurl'])

    year_dir = os.path.join(args.image_dir, str(year))
    os.makedirs(year_dir, exist_ok=True)
    out_path = os.path.join(year_dir, f'{photo_id}.jpg')

    if os.path.exists(out_path) and os.path.getsize(out_path) > args.min_kb * 1024:
        return ('skip', photo_id, year, month, out_path)

    # try a few size variants of the Flickr URL
    base, _, ext = url.rpartition('.')
    ext = ext or 'jpg'
    urls_to_try = [f'{base}_z.{ext}', url, f'{base}_m.{ext}']

    for attempt_url in urls_to_try:
        try:
            r = requests.get(attempt_url, timeout=args.timeout,
                             headers={'User-Agent': 'Mozilla/5.0'})
            if r.status_code != 200 or len(r.content) < args.min_kb * 1024:
                continue
            img = Image.open(BytesIO(r.content))
            img.verify()
            with open(out_path, 'wb') as f:
                f.write(r.content)
            return ('ok', photo_id, year, month, out_path)
        except Exception:
            time.sleep(0.3)

    return ('failed', photo_id, year, month, None)


rows = df_all.to_dict('records')
results = {'ok': 0, 'skip': 0, 'failed': 0}
manifest_rows = []

# pick up where we left off if the task manifest already exists
if os.path.exists(task_manifest):
    existing = pd.read_csv(task_manifest)
    manifest_rows = existing.to_dict('records')
    print(f"resuming - found {len(manifest_rows):,} existing entries")

print(f"\ndownloading {len(rows):,} images with {args.n_workers} threads")
print("safe to kill and resubmit\n")

chunk_start = 0
while chunk_start < len(rows):
    chunk = rows[chunk_start : chunk_start + args.chunk_size]

    with ThreadPoolExecutor(max_workers=args.n_workers) as executor:
        futures = {executor.submit(download_one, r): r for r in chunk}
        with tqdm(total=len(chunk), desc=f"chunk {chunk_start // args.chunk_size + 1}") as pbar:
            for future in as_completed(futures):
                status, photo_id, year, month, path = future.result()
                results[status if status in results else 'failed'] += 1
                if path:
                    manifest_rows.append({
                        'photo_id': photo_id,
                        'year':     year,
                        'month':    month,
                        'path':     path,
                    })
                pbar.set_postfix(results)
                pbar.update(1)

    pd.DataFrame(manifest_rows).to_csv(task_manifest, index=False)
    print(f"  flushed manifest: {len(manifest_rows):,} rows | {results}")
    chunk_start += args.chunk_size

print(f"\nfinal counts: {results}")

if args.num_tasks == 1:
    df_manifest = pd.DataFrame(manifest_rows)
    df_manifest.to_csv(args.manifest_out, index=False)
    print(f"\nmanifest saved: {len(df_manifest):,} images")
    print(df_manifest.groupby('year').size().to_string())
else:
    print(f"\ntask {args.task_id} done - partial manifest at {task_manifest}")
    print("run merge_manifests.py once all tasks complete")
