"""
Step 3 of the YFCC-DINO pipeline.

Extracts 384-dimensional DINO ViT-S/16 embeddings from the downloaded
YFCC images, matching the feature extractor used in the DeDrift paper.

I ran this on the HPC cluster I have access to via RACKlette

Output:
    data/yfcc_sampled/embeddings/{year}_embeddings.npy  — float32 (N, 384)
    data/yfcc_sampled/embeddings/{year}_photo_ids.npy   — str array (N,)
    data/yfcc_sampled/embeddings/metadata.csv           — photo_id, year, month

Post-processing (matching DeDrift):
    - Apply random rotation to all embeddings
    - Quantize to 8 bits (rescale to uint8)
    - Save final combined arrays

Requirements:
    pip install torch torchvision timm tqdm pandas pillow numpy
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageFile
from tqdm import tqdm
import argparse
import warnings
warnings.filterwarnings('ignore')
#allow truncated images as long as they contain enough information to decode something meaningful
ImageFile.LOAD_TRUNCATED_IMAGES = True

# parameters
parser = argparse.ArgumentParser(description='Extract DINO embeddings')
parser.add_argument('--image_dir',   default='data/yfcc_sampled/images')
parser.add_argument('--manifest',    default='data/yfcc_sampled/downloaded_manifest.csv')
parser.add_argument('--output_dir',  default='data/yfcc_sampled/embeddings')
parser.add_argument('--batch_size',  type=int, default=256)
parser.add_argument('--num_workers', type=int, default=8)
args = parser.parse_args()

IMAGE_DIR    = args.image_dir
MANIFEST_CSV = args.manifest
OUTPUT_DIR   = args.output_dir
BATCH_SIZE   = args.batch_size
NUM_WORKERS  = args.num_workers
YEARS        = list(range(2007, 2014))
RANDOM_SEED  = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

print(f"image_dir:   {IMAGE_DIR}")
print(f"manifest:    {MANIFEST_CSV}")
print(f"output_dir:  {OUTPUT_DIR}")
print(f"batch_size:  {BATCH_SIZE}")
print(f"num_workers: {NUM_WORKERS}")

# device
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    device = torch.device('mps')
    print("Using Apple MPS")
else:
    device = torch.device('cpu')
    print("Using CPU (slow — consider using HPC GPU)")

# load DINO ViT-S/16 
print("\nLoading DINO ViT-S/16 model...")
model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16',
                       pretrained=True)
model = model.to(device)
model.eval()
print(f"  Model loaded, embedding dim = 384")

# image transforms (standard DINO preprocessing)
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# dataset
class YFCCDataset(Dataset):
    def __init__(self, paths, photo_ids, transform):
        self.paths     = paths
        self.photo_ids = photo_ids
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        try:
            img = Image.open(path).convert('RGB')
            return self.transform(img), idx
        except Exception:
            # return blank image for broken files
            return torch.zeros(3, 224, 224), idx

# load manifest
df_manifest = pd.read_csv(MANIFEST_CSV)
print(f"\nLoaded manifest: {len(df_manifest):,} images")
print(df_manifest.groupby('year').size().to_string())

# extract embeddings per year 
all_embeddings = []
all_photo_ids  = []
all_years      = []
all_months     = []

for year in YEARS:
    year_df   = df_manifest[df_manifest['year'] == year].reset_index(drop=True)
    out_emb   = os.path.join(OUTPUT_DIR, f'{year}_embeddings.npy')
    out_ids   = os.path.join(OUTPUT_DIR, f'{year}_photo_ids.npy')

    if os.path.exists(out_emb) and os.path.exists(out_ids):
        print(f"\nYear {year}: loading cached embeddings...")
        embs = np.load(out_emb)
        ids  = np.load(out_ids)
        all_embeddings.append(embs)
        all_photo_ids.extend(ids.tolist())
        all_years.extend([year] * len(embs))
        continue

    print(f"\nYear {year}: extracting embeddings for {len(year_df):,} images...")

    paths     = year_df['path'].tolist()
    photo_ids = year_df['photo_id'].astype(str).tolist()

    dataset    = YFCCDataset(paths, photo_ids, transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE,
                            num_workers=NUM_WORKERS, pin_memory=True,
                            shuffle=False)

    year_embeddings = np.zeros((len(dataset), 384), dtype='float32')
    year_ids        = [''] * len(dataset)

    with torch.no_grad():
        for batch_imgs, batch_indices in tqdm(dataloader,
                                               desc=f'  {year}',
                                               leave=False):
            batch_imgs = batch_imgs.to(device)
            # DINO forward pass — returns CLS token (384-dim)
            embs = model(batch_imgs).cpu().numpy()
            for i, idx in enumerate(batch_indices.numpy()):
                year_embeddings[idx] = embs[i]
                year_ids[idx]        = photo_ids[idx]

    # save per-year arrays
    np.save(out_emb, year_embeddings)
    np.save(out_ids, np.array(year_ids))
    print(f"  Saved {len(year_embeddings):,} embeddings for {year}")

    all_embeddings.append(year_embeddings)
    all_photo_ids.extend(year_ids)
    all_years.extend([year] * len(year_embeddings))

# combine all years
print("\nCombining all years...")
embeddings_all = np.vstack(all_embeddings)  # (N_total, 384)
print(f"  Total embeddings: {embeddings_all.shape}")

# post-processing: random rotation (matching DeDrift)
print("Applying random rotation...")
np.random.seed(RANDOM_SEED)
# random orthogonal matrix via QR decomposition
rand_mat  = np.random.randn(384, 384).astype('float32')
Q, _      = np.linalg.qr(rand_mat)
embeddings_rotated = embeddings_all @ Q
print(f"  Rotation applied, shape: {embeddings_rotated.shape}")

# post-processing: 8-bit quantization (matching DeDrift)
print("Quantizing to 8 bits...")
# normalize to [0, 1] then scale to [0, 255]
emb_min  = embeddings_rotated.min(axis=0, keepdims=True)
emb_max  = embeddings_rotated.max(axis=0, keepdims=True)
emb_norm = (embeddings_rotated - emb_min) / (emb_max - emb_min + 1e-8)
embeddings_uint8 = (emb_norm * 255).astype('uint8')

# also save float32 version for experiments (before quantization)
embeddings_float32 = embeddings_rotated.astype('float32')

# save final combined dataset
print("\nSaving final dataset...")

# save rotation matrix so we can apply it to new queries consistently
np.save(os.path.join(OUTPUT_DIR, 'rotation_matrix.npy'), Q)
np.save(os.path.join(OUTPUT_DIR, 'quant_min.npy'), emb_min)
np.save(os.path.join(OUTPUT_DIR, 'quant_max.npy'), emb_max)

# float32 version (for use with hnswlib which needs float32)
np.save(os.path.join(OUTPUT_DIR, 'embeddings_float32.npy'), embeddings_float32)

# metadata: photo_id, year, month
df_meta = pd.read_csv(MANIFEST_CSV)
df_meta['photo_id'] = df_meta['photo_id'].astype(str)
df_ordered = pd.DataFrame({
    'photo_id': [str(p) for p in all_photo_ids],
    'year':     all_years,
})
df_ordered = df_ordered.merge(
    df_meta[['photo_id', 'year', 'month']] if 'month' in df_meta.columns
    else df_meta[['photo_id', 'year']],
    on=['photo_id', 'year'], how='left'
)
df_ordered.to_csv(os.path.join(OUTPUT_DIR, 'metadata.csv'), index=False)

print(f"\nFinal dataset saved to {OUTPUT_DIR}/")
print(f"  embeddings_float32.npy : {embeddings_float32.shape}, float32")
print(f"  metadata.csv           : {len(df_ordered):,} rows")
print(f"\nYear distribution:")
print(df_ordered.groupby('year').size().to_string())
print("\nStep 3 complete.")