"""
Step 2 of the YFCC-DINO pipeline.

Downloads images from the Flickr URLs sampled in step 1.

Requirements:
    pip install requests tqdm pandas pillow
"""

import os
import requests
import pandas as pd
from tqdm import tqdm
from PIL import Image, ImageFile
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# allow truncated images as long as there is enough information to decode something meaningful
ImageFile.LOAD_TRUNCATED_IMAGES = True

# parameters
SAMPLED_CSV  = 'data/yfcc_sampled/sampled_urls.csv'
IMAGE_DIR    = 'data/yfcc_sampled/images'
N_WORKERS    = 32
TIMEOUT      = 15
MIN_KB       = 5

os.makedirs(IMAGE_DIR, exist_ok=True)

df = pd.read_csv(SAMPLED_CSV)
print(f"Loaded {len(df):,} URLs")
print(df.groupby('year').size().to_string())

def download_one(row):
    year     = int(row['year'])
    photo_id = str(row['photoid'])
    url      = str(row['downloadurl'])

    year_dir = os.path.join(IMAGE_DIR, str(year))
    os.makedirs(year_dir, exist_ok=True)
    out_path = os.path.join(year_dir, f'{photo_id}.jpg')

    if os.path.exists(out_path) and os.path.getsize(out_path) > MIN_KB * 1024:
        return 'skip'

    # try original URL and size variants
    base_parts = url.rsplit('.', 1)
    ext = base_parts[-1] if len(base_parts) > 1 else 'jpg'
    base = base_parts[0]
    urls_to_try = [f"{base}_z.{ext}", url, f"{base}_m.{ext}"]

    for attempt_url in urls_to_try:
        try:
            r = requests.get(attempt_url, timeout=TIMEOUT,
                             headers={'User-Agent': 'Mozilla/5.0'})
            if r.status_code != 200:
                continue
            if len(r.content) < MIN_KB * 1024:
                continue
            img = Image.open(BytesIO(r.content))
            img.verify()
            with open(out_path, 'wb') as f:
                f.write(r.content)
            return 'ok'
        except Exception:
            time.sleep(0.5)
            continue
    return 'failed'

rows    = df.to_dict('records')
results = {'ok': 0, 'skip': 0, 'failed': 0}

print(f"\nDownloading {len(rows):,} images ({N_WORKERS} threads)...")
print("Safe to Ctrl+C and resume\n")

with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
    futures = {executor.submit(download_one, row): row for row in rows}
    with tqdm(total=len(rows)) as pbar:
        for future in as_completed(futures):
            result = future.result()
            key    = result if result in results else 'failed'
            results[key] += 1
            pbar.set_postfix(results)
            pbar.update(1)

print(f"\nResults: {results}")

# save manifest
manifest = []
for year in range(2007, 2014):
    year_dir = os.path.join(IMAGE_DIR, str(year))
    if not os.path.exists(year_dir):
        continue
    for fname in os.listdir(year_dir):
        if fname.endswith('.jpg'):
            manifest.append({
                'photo_id': fname.replace('.jpg', ''),
                'year':     year,
                'path':     os.path.join(year_dir, fname),
            })

df_manifest = pd.DataFrame(manifest)
manifest_path = 'data/yfcc_sampled/downloaded_manifest.csv'
df_manifest.to_csv(manifest_path, index=False)
print(f"\nManifest: {len(df_manifest):,} images")
print(df_manifest.groupby('year').size().to_string())
print("\nStep 2 complete.")