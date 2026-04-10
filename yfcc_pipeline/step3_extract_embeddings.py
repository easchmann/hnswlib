"""
Extracts 384-dim DINO ViT-S/16 embeddings from downloaded YFCC images,
matching the feature extractor used in the DeDrift paper.

To avoid OOM on large datasets, images are processed in shards per year.
Each shard is saved immediately, so the job can be safely requeued - already
completed shards are skipped. After all shards are done, the per-year files
are merged, the random rotation and quantization are applied, and the shard
intermediates are cleaned up to save space. The raw images are also deleted
at the end since we only need the embeddings going forward.

Output:
    data/yfcc_sampled/embeddings/embeddings_float32.npy  - (N, 384) memory-mapped
    data/yfcc_sampled/embeddings/metadata.csv            - photo_id, year, month
    data/yfcc_sampled/embeddings/rotation_matrix.npy
    data/yfcc_sampled/embeddings/quant_min.npy / quant_max.npy

Requirements:
    pip install torch torchvision tqdm pandas pillow numpy
"""

import os
import shutil
import argparse
import warnings
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageFile
from tqdm import tqdm

warnings.filterwarnings('ignore')
ImageFile.LOAD_TRUNCATED_IMAGES = True

parser = argparse.ArgumentParser()
parser.add_argument('--image_dir',   default='data/yfcc_sampled/images')
parser.add_argument('--manifest',    default='data/yfcc_sampled/downloaded_manifest.csv')
parser.add_argument('--output_dir',  default='data/yfcc_sampled/embeddings')
parser.add_argument('--batch_size',  type=int, default=256)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--shard_size',  type=int, default=200_000)
parser.add_argument('--years',       nargs='+', type=int, default=list(range(2007, 2014)))
args = parser.parse_args()

SHARD_DIR   = os.path.join(args.output_dir, 'shards')
RANDOM_SEED = 42

os.makedirs(SHARD_DIR, exist_ok=True)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

print(f"image_dir:   {args.image_dir}")
print(f"manifest:    {args.manifest}")
print(f"output_dir:  {args.output_dir}")
print(f"batch_size:  {args.batch_size}, num_workers: {args.num_workers}, shard_size: {args.shard_size:,}")
print(f"years:       {args.years}")

if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    device = torch.device('mps')
    print("\nApple MPS")
else:
    device = torch.device('cpu')
    print("\nCPU (this will be slow)")

print("loading DINO ViT-S/16...")
model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16', pretrained=True)
model = model.to(device)
model.eval()
print("  embedding dim: 384")

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


class YFCCDataset(Dataset):
    def __init__(self, paths, photo_ids, transform):
        self.paths     = paths
        self.photo_ids = photo_ids
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.paths[idx]).convert('RGB')
            return self.transform(img), idx
        except Exception:
            return torch.zeros(3, 224, 224), idx


def extract_shards(year, year_df):
    paths     = year_df['path'].tolist()
    photo_ids = year_df['photo_id'].astype(str).tolist()
    n         = len(paths)

    emb_files = []
    ids_files = []

    for shard_idx, start in enumerate(range(0, n, args.shard_size)):
        end        = min(start + args.shard_size, n)
        shard_size = end - start

        out_emb = os.path.join(SHARD_DIR, f'{year}_shard{shard_idx:04d}_embeddings.npy')
        out_ids = os.path.join(SHARD_DIR, f'{year}_shard{shard_idx:04d}_photo_ids.npy')

        if os.path.exists(out_emb) and os.path.exists(out_ids):
            print(f"  shard {shard_idx}: cached, skipping")
            emb_files.append(out_emb)
            ids_files.append(out_ids)
            continue

        print(f"  shard {shard_idx}: images {start:,}-{end:,}")

        dataset    = YFCCDataset(paths[start:end], photo_ids[start:end], transform)
        dataloader = DataLoader(dataset, batch_size=args.batch_size,
                                num_workers=args.num_workers, pin_memory=True,
                                shuffle=False)

        shard_embs = np.zeros((shard_size, 384), dtype='float32')
        shard_ids  = [''] * shard_size

        with torch.no_grad():
            for imgs, indices in tqdm(dataloader, desc=f'    shard {shard_idx}', leave=False):
                embs = model(imgs.to(device)).cpu().numpy()
                for i, idx in enumerate(indices.numpy()):
                    shard_embs[idx] = embs[i]
                    shard_ids[idx]  = photo_ids[start + idx]

        np.save(out_emb, shard_embs)
        np.save(out_ids, np.array(shard_ids))
        print(f"    saved {shard_size:,} embeddings")

        emb_files.append(out_emb)
        ids_files.append(out_ids)

    return emb_files, ids_files


def merge_year(year, emb_files, ids_files):
    out_emb = os.path.join(args.output_dir, f'{year}_embeddings.npy')
    out_ids = os.path.join(args.output_dir, f'{year}_photo_ids.npy')

    if os.path.exists(out_emb) and os.path.exists(out_ids):
        print(f"  {year}: merged file already exists, loading")
        return np.load(out_emb), np.load(out_ids).tolist()

    embs = np.vstack([np.load(f) for f in emb_files])
    ids  = [pid for f in ids_files for pid in np.load(f).tolist()]

    np.save(out_emb, embs)
    np.save(out_ids, np.array(ids))
    print(f"  {year}: merged {len(embs):,} embeddings")
    return embs, ids


# main loop
df_manifest = pd.read_csv(args.manifest)
print(f"\nmanifest: {len(df_manifest):,} images")
print(df_manifest.groupby('year').size().to_string())

all_embeddings = []
all_photo_ids  = []
all_years      = []

for year in args.years:
    year_df = df_manifest[df_manifest['year'] == year].reset_index(drop=True)
    print(f"\nyear {year}: {len(year_df):,} images")
    emb_files, ids_files = extract_shards(year, year_df)
    embs, ids            = merge_year(year, emb_files, ids_files)
    all_embeddings.append(embs)
    all_photo_ids.extend(ids)
    all_years.extend([year] * len(embs))

print(f"\ncombining {len(all_years):,} embeddings across all years...")
embeddings_all = np.vstack(all_embeddings)

# random rotation (matching DeDrift preprocessing)
rot_path = os.path.join(args.output_dir, 'rotation_matrix.npy')
if os.path.exists(rot_path):
    Q = np.load(rot_path)
    print("loaded existing rotation matrix")
else:
    np.random.seed(RANDOM_SEED)
    Q, _ = np.linalg.qr(np.random.randn(384, 384).astype('float32'))
    np.save(rot_path, Q)
    print("generated random rotation matrix")

# apply rotation in chunks and write directly to a memory-mapped file
# avoids keeping two full copies of the embedding array in RAM at once
print("applying rotation (chunked to save RAM)...")
out_f32 = os.path.join(args.output_dir, 'embeddings_float32.npy')
fp = np.lib.format.open_memmap(out_f32, mode='w+', dtype='float32', shape=embeddings_all.shape)
chunk = 500_000
for i in range(0, len(embeddings_all), chunk):
    fp[i:i+chunk] = embeddings_all[i:i+chunk] @ Q
fp.flush()
del embeddings_all

# quantization range (saved for applying to query embeddings later)
emb_min = fp.min(axis=0, keepdims=True)
emb_max = fp.max(axis=0, keepdims=True)
np.save(os.path.join(args.output_dir, 'quant_min.npy'), emb_min)
np.save(os.path.join(args.output_dir, 'quant_max.npy'), emb_max)

# metadata
df_meta = pd.read_csv(args.manifest)
df_meta['photo_id'] = df_meta['photo_id'].astype(str)
merge_cols = ['photo_id', 'year'] + (['month'] if 'month' in df_meta.columns else [])

df_ordered = pd.DataFrame({'photo_id': [str(p) for p in all_photo_ids], 'year': all_years})
df_ordered = df_ordered.merge(df_meta[merge_cols], on=['photo_id', 'year'], how='left')
df_ordered.to_csv(os.path.join(args.output_dir, 'metadata.csv'), index=False)

print(f"\nfinal dataset:")
print(f"  embeddings_float32.npy: {fp.shape}, float32")
print(f"  metadata.csv: {len(df_ordered):,} rows")
print(df_ordered.groupby('year').size().to_string())

# clean up shard intermediates - no longer needed now that per-year files are merged
print("\ncleaning up shards...")
shutil.rmtree(SHARD_DIR)
print(f"  removed {SHARD_DIR}")

# delete raw images - we only need the embeddings from here on
# this frees ~320GB on scratch
print("deleting raw images to free disk space...")
shutil.rmtree(args.image_dir)
print(f"  removed {args.image_dir}")
print("  (re-run step2 if you need them back)")

print("\ndone.")
