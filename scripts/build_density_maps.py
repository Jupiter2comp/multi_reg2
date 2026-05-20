"""
Build gene expression density maps from 10x Visium data.
Reads .mtx, .tsv files directly without scanpy dependency.

Used to preprocess the public datasets described in manuscript Section 6.2:
  - Human Breast Cancer: Visium Fresh Frozen, Whole Transcriptome
    (spaceranger 1.3.0, 4,898 tissue spots, 0.3115 um/px)
  - Human Brain Cancer, 11 mm Capture Area (FFPE)
    (spaceranger 2.0.1, 10,878 tissue spots, 0.251 um/px)

Output: .npz files with hires image, density maps, and pixel size metadata.
"""
import os, gzip
import numpy as np
import pandas as pd
from scipy.io import mmread
from PIL import Image
import json


def load_visium_data(data_dir):
    """Load Visium data and return expression matrix, barcodes, and feature names."""
    mtx_path = barcodes_path = features_path = None
    bc_dir = os.path.join(data_dir, 'filtered_feature_bc_matrix')
    for f in os.listdir(bc_dir):
        if f.endswith('matrix.mtx.gz') or f.endswith('matrix.mtx'):
            mtx_path = os.path.join(bc_dir, f)
        elif 'barcodes' in f and (f.endswith('.tsv.gz') or f.endswith('.tsv')):
            barcodes_path = os.path.join(bc_dir, f)
        elif 'features' in f and (f.endswith('.tsv.gz') or f.endswith('.tsv')):
            features_path = os.path.join(bc_dir, f)

    if mtx_path.endswith('.gz'):
        with gzip.open(mtx_path, 'rt') as f:
            X = mmread(f).tocsr()
    else:
        X = mmread(mtx_path).tocsr()

    if barcodes_path.endswith('.gz'):
        barcodes = pd.read_csv(barcodes_path, header=None, compression='gzip')[0].tolist()
    else:
        barcodes = pd.read_csv(barcodes_path, header=None)[0].tolist()

    if features_path.endswith('.gz'):
        features = pd.read_csv(features_path, header=None, sep='\t', compression='gzip')
    else:
        features = pd.read_csv(features_path, header=None, sep='\t')
    gene_names = features[1].tolist() if features.shape[1] >= 2 else features[0].tolist()
    return X, barcodes, gene_names


def load_spatial_positions(data_dir):
    """Load tissue positions CSV."""
    for f in ['tissue_positions_list.csv', 'tissue_positions.csv']:
        p = os.path.join(data_dir, 'spatial', f)
        if os.path.exists(p):
            with open(p) as fh:
                first_line = fh.readline().strip()
            has_header = first_line.startswith('barcode')
            positions = pd.read_csv(p, header=0 if has_header else None)
            cols = ['barcode', 'in_tissue', 'array_row', 'array_col', 'pxl_row_in_fullres', 'pxl_col_in_fullres']
            if not has_header:
                positions.columns = cols
            else:
                positions.columns = cols
            return positions
    raise FileNotFoundError("No tissue position file found in spatial/")


def build_density_map(pos_df, value_col, w, h, spot_diam_fullres, scale):
    """Build a density map by rendering spots as circles."""
    canvas = np.zeros((h, w), dtype=np.float32)
    spot_r = max(1, int(spot_diam_fullres * scale / 2))
    for _, row in pos_df.iterrows():
        cx = int(row['px_hires'] * scale)
        cy = int(row['py_hires'] * scale)
        val = row[value_col]
        if val <= 0:
            continue
        val_norm = min(val, np.percentile(pos_df[value_col], 95))
        for dy in range(-spot_r, spot_r + 1):
            for dx in range(-spot_r, spot_r + 1):
                if dx*dx + dy*dy <= spot_r*spot_r:
                    px, py = cx + dx, cy + dy
                    if 0 <= px < w and 0 <= py < h:
                        canvas[py, px] += val_norm
    if canvas.max() > 0:
        canvas /= np.percentile(canvas, 99.5)
        canvas = np.clip(canvas, 0, 1)
    return canvas


def process_dataset(data_dir, output_path):
    """Process a single Visium dataset and save density maps."""
    print(f"\nLoading data from {data_dir}...")

    # Load expression matrix
    X, barcodes, gene_names = load_visium_data(data_dir)
    print(f"  Matrix shape: {X.shape[0]} genes x {X.shape[1]} spots")

    # Load positions
    positions = load_spatial_positions(data_dir)
    tissue = positions[positions['in_tissue'] == 1].copy()
    print(f"  Tissue spots: {len(tissue)}")

    # Match barcodes
    barcode_to_idx = {b: i for i, b in enumerate(barcodes)}
    matched = tissue[tissue['barcode'].isin(barcode_to_idx)]
    col_indices = [barcode_to_idx[b] for b in matched['barcode'].values]
    X_tissue = X[:, col_indices]
    total_counts = np.array(X_tissue.sum(axis=0)).flatten()
    matched['total_counts'] = total_counts

    # Load scalefactors
    with open(os.path.join(data_dir, 'spatial', 'scalefactors_json.json')) as f:
        sf = json.load(f)
    spot_diam = sf['spot_diameter_fullres']
    scale = sf['tissue_hires_scalef']
    pixel_size_um = 55.0 / spot_diam

    # Load hires image
    hires = np.array(Image.open(os.path.join(data_dir, 'spatial', 'tissue_hires_image.png'))) / 255.0

    # Convert positions to hires coords
    matched['px_hires'] = matched['pxl_col_in_fullres'] * scale
    matched['py_hires'] = matched['pxl_row_in_fullres'] * scale

    # Build density maps
    h, w = hires.shape[:2]
    density_total = build_density_map(matched, 'total_counts', w, h, spot_diam, scale)
    matched['mean_expr'] = total_counts / (X_tissue.shape[0] + 1)
    density_mean = build_density_map(matched, 'mean_expr', w, h, spot_diam, scale)

    # Save
    np.savez_compressed(output_path,
        hires_img=hires,
        density_total=density_total,
        density_mean=density_mean,
        pixel_size_um=pixel_size_um)
    print(f"  Saved to {output_path}")
    print(f"  Hires: {hires.shape}, pixel_size: {pixel_size_um:.4f} um/px")
    print(f"  Tissue spots: {len(matched)}")

    return hires, density_total, pixel_size_um


if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    data_root = os.environ.get('VISIUM_DATA_ROOT',
        os.path.join(repo_root, 'data', 'raw'))
    out_root = os.environ.get('VISIUM_OUT_ROOT',
        os.path.join(repo_root, 'data', 'processed'))

    datasets = {
        'breast': {
            'dir': os.path.join(data_root, 'Visium_Fresh_Frozen_Human_Breast_Cancer'),
            'out': os.path.join(out_root, 'breast_density_maps.npz')
        },
        'brain': {
            'dir': os.path.join(data_root, 'Visium_FFPE_Human_Brain_Cancer_11mm'),
            'out': os.path.join(out_root, 'brain_density_maps.npz')
        }
    }

    for name, cfg in datasets.items():
        if os.path.exists(cfg['dir']):
            process_dataset(cfg['dir'], cfg['out'])
        else:
            print(f"  Skipping {name}: {cfg['dir']} not found")
