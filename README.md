# Hierarchical Registration for Spatial Omics

Code for hierarchical image registration pipeline: rigid pre-alignment, affine, and B-spline free-form deformation.

## Datasets

- **Human Breast Cancer: Visium Fresh Frozen, Whole Transcriptome** (4,898 spots, 0.3115 µm/px) — [10x Genomics](https://www.10xgenomics.com/datasets/human-breast-cancer-visium-fresh-frozen-whole-transcriptome-1-standard)
- **Human Brain Cancer, 11 mm Capture Area (FFPE)** (10,878 spots, 0.251 µm/px) — [10x Genomics](https://www.10xgenomics.com/datasets/human-brain-cancer-11-mm-capture-area-ffpe-2-standard)

Download spaceranger output to `data/raw/`.

## Usage

```bash
pip install -r requirements.txt
python scripts/build_density_maps.py   # preprocess Visium data
python reproduce_results.py            # run all experiments
```

## Structure

```
├── reproduce_results.py    # end-to-end reproduction
├── src/registration_pipeline.py  # core methods (SimpleITK)
├── scripts/
│   ├── build_density_maps.py   # Visium → density maps
│   ├── data_generation.py      # synthetic IHC data
│   └── evaluate.py             # CLI evaluation
├── config/                     # elastix reference configs
└── results/                    # pre-computed figures and tables
```
