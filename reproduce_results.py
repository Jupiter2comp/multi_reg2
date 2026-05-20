#!/usr/bin/env python3
"""
End-to-end reproduction of all manuscript results.
Usage: python reproduce_results.py [--no-synthetic] [--no-real] [--grid G20 G30 G40]

Generates:
  - results/results.json: all TRE and NCC metrics
  - results/figures/: comparison and ablation plots
  - results/tables/: LaTeX tables for manuscript
"""
import os, sys, json, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from src.registration_pipeline import (
    apply_synthetic_deformation,
    compute_target_registration_error,
    rigid_registration,
    affine_registration,
    bspline_registration,
    demons_registration,
    ncc
)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'results')
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
FIGURES_DIR = os.path.join(OUTPUT_DIR, 'figures')
TABLES_DIR = os.path.join(OUTPUT_DIR, 'tables')
GRID_SPACINGS = [20, 30, 40]
PIXEL_SIZE = {'breast': 0.3115, 'brain': 0.2510}


def ensure_dirs():
    for d in [OUTPUT_DIR, FIGURES_DIR, TABLES_DIR, DATA_DIR]:
        os.makedirs(d, exist_ok=True)


def figure_synthetic_comparison(fixed, moving, registered_dict, disp_r, disp_c,
                                 gt_affine, save_path):
    """Figure 1: Synthetic IHC registration comparison (manuscript Fig 1)."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    titles = ['Fixed (IHC)', 'Moving (Deformed)', 'Rigid', 'Affine',
              'B-Spline (GS=30)', 'SyN-Demons',
              'Moving Overlay', 'B-Spline Overlay']

    overlap_moving = np.abs(fixed - moving)
    overlap_bspline = np.abs(fixed - registered_dict.get('bspline', moving))

    images = [fixed, moving,
              registered_dict.get('rigid', fixed),
              registered_dict.get('affine', fixed),
              registered_dict.get('bspline', fixed),
              registered_dict.get('demons', fixed),
              overlap_moving, overlap_bspline]

    for idx, ax in enumerate(axes.flat):
        if idx < len(images):
            im = ax.imshow(images[idx], cmap='gray' if images[idx].ndim == 2 else None)
            ax.set_title(titles[idx])
            ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved {save_path}")


def figure_real_comparison(fixed, moving, registered_dict, dataset_name, save_path):
    """Figure 2: Real Visium registration comparison (manuscript Fig 2)."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    titles = ['Fixed (' + dataset_name + ')', 'Moving (Deformed)', 'Rigid', 'Affine',
              'B-Spline (GS=30)', 'SyN-Demons',
              'Moving Overlay', 'B-Spline Overlay']

    overlap_moving = np.abs(fixed - moving)
    overlap_bspline = np.abs(fixed - registered_dict.get('bspline', moving))

    images = [fixed, moving,
              registered_dict.get('rigid', fixed),
              registered_dict.get('affine', fixed),
              registered_dict.get('bspline', fixed),
              registered_dict.get('demons', fixed),
              overlap_moving, overlap_bspline]

    for idx, ax in enumerate(axes.flat):
        im = ax.imshow(images[idx], cmap='magma')
        ax.set_title(titles[idx])
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved {save_path}")


def figure_ablation(fixed, moving, results, save_path):
    """Figure 3: Ablation study bar chart (manuscript Fig 3)."""
    methods = list(results.keys())
    tre_means = [results[m].get('tre_mean_um', results[m].get('tre_mean_px', 0))
                 for m in methods]
    tre_stds = [results[m].get('tre_std_um', results[m].get('tre_std_px', 0))
                for m in methods]
    colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2']

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(methods))
    bars = ax.bar(x, tre_means, yerr=tre_stds, capsize=5, color=colors, width=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([m.capitalize() for m in methods])
    ax.set_ylabel('TRE (μm)' if 'um' in str(type(tre_means[0])) else 'TRE (px)')
    ax.set_title('Registration Method Comparison')
    ax.grid(axis='y', alpha=0.3)

    for bar, val in zip(bars, tre_means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved {save_path}")


def generate_table(results, save_path, title='Registration Results'):
    """Generate LaTeX table of registration metrics."""
    lines = [
        r'\begin{table}[htbp]',
        r'\centering',
        r'\caption{' + title + '}',
        r'\label{tab:registration_comparison}',
        r'\begin{tabular}{lcccccc}',
        r'\toprule',
        r'Method & TRE (μm) & NCC & Time (s) & GS \\',
        r'\midrule'
    ]

    for method, metrics in results.items():
        if 'error' in metrics:
            continue
        tre = format(metrics.get('tre_mean_um', metrics.get('tre_mean_px', 0)), '.4f')
        ncc_val = format(metrics.get('ncc', 0), '.4f')
        t = format(metrics.get('time_s', 0), '.1f')
        gs = metrics.get('grid_spacing', 'N/A')
        lines.append(f'{method.capitalize()} & {tre} & {ncc_val} & {t} & {gs} \\\\')

    lines.extend([
        r'\bottomrule',
        r'\end{tabular}',
        r'\end{table}'
    ])

    with open(save_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"  Saved {save_path}")


def run_synthetic():
    print("\n" + "=" * 60)
    print("SYNTHETIC EXPERIMENT")
    print("=" * 60)

    from skimage import data, io
    fixed = data.immunohistochemistry()
    if fixed.max() > 1.0:
        fixed = fixed / 255.0
    fixed_gray = fixed.mean(axis=2) if fixed.ndim == 3 else fixed

    print(f"  IHC image: {fixed.shape}")
    moving, (gt_affine, disp_r, disp_c) = apply_synthetic_deformation(fixed)
    moving_gray = moving.mean(axis=2) if moving.ndim == 3 else moving

    io.imsave(os.path.join(DATA_DIR, 'fixed.png'), (fixed * 255).astype(np.uint8))
    io.imsave(os.path.join(DATA_DIR, 'moving.png'), (moving * 255).astype(np.uint8))
    np.save(os.path.join(DATA_DIR, 'gt_displacement_x.npy'), disp_r)
    np.save(os.path.join(DATA_DIR, 'gt_displacement_y.npy'), disp_c)

    methods = ['rigid', 'affine', 'bspline', 'demons']
    reg_map = {}
    for m in methods:
        print(f"  Running {m}...")
        if m == 'rigid':
            r, meta = rigid_registration(fixed_gray, moving_gray)
        elif m == 'affine':
            r, meta = affine_registration(fixed_gray, moving_gray)
        elif m == 'bspline':
            best_r, best_meta = None, {}
            best_tre = float('inf')
            for gs in GRID_SPACINGS:
                r2, m2 = bspline_registration(fixed_gray, moving_gray, grid_spacing=gs)
                if r2 is not None:
                    tre2, _, _, _, _ = compute_target_registration_error(
                        fixed_gray, moving_gray, r2, None, disp_r, disp_c)
                    if tre2 < best_tre:
                        best_tre = tre2
                        best_r, best_meta = r2, m2
            r, meta = best_r, best_meta
        elif m == 'demons':
            r, meta = demons_registration(fixed_gray, moving_gray)
        reg_map[m] = r if r is not None else moving_gray

    results = {}
    for m in methods:
        r = reg_map[m]
        tre_m, tre_s, tre_med, tre_max, _ = compute_target_registration_error(
            fixed_gray, moving_gray, r, None, disp_r, disp_c)
        results[m] = {
            'tre_mean_px': float(tre_m), 'tre_std_px': float(tre_s),
            'tre_median_px': float(tre_med), 'tre_max_px': float(tre_max),
            'ncc': float(ncc(fixed_gray, r))
        }
        print(f"    TRE: {tre_m:.4f}+-{tre_s:.4f} px, NCC: {results[m]['ncc']:.4f}")

    figure_synthetic_comparison(fixed_gray, moving_gray, reg_map, disp_r, disp_c,
                                 gt_affine, os.path.join(FIGURES_DIR, 'synthetic_comparison.png'))
    figure_ablation(fixed_gray, reg_map, results,
                     os.path.join(FIGURES_DIR, 'synthetic_ablation.png'))
    generate_table(results, os.path.join(TABLES_DIR, 'synthetic_ablation.tex'),
                    'Synthetic IHC Registration Results')
    return results


def run_real():
    print("\n" + "=" * 60)
    print("REAL VISIUM EXPERIMENT")
    print("=" * 60)

    data_root = os.path.join(os.path.dirname(__file__), 'data', 'processed')
    all_results = {}

    for dataset_name in ['breast', 'brain']:
        npz_path = os.path.join(data_root, f'{dataset_name}_density_maps.npz')
        if not os.path.exists(npz_path):
            print(f"\n  {dataset_name}: density maps not found at {npz_path}")
            # Fall back to experiment_results_v3
            fallback = os.path.join(os.path.dirname(__file__), '..', 'experiment_results_v3',
                                    'processed_data', f'{dataset_name}_density_maps.npz')
            if os.path.exists(fallback):
                npz_path = fallback
            else:
                print(f"    Skipping {dataset_name} (no data)")
                continue

        data = np.load(npz_path)
        fixed = data['density_total']
        scale = float(data['pixel_size_um'].item() if data['pixel_size_um'].ndim == 0 else data['pixel_size_um'][0])
        print(f"\n  {dataset_name}: {fixed.shape}, {scale:.4f} um/px")

        moving, (aff, disp_r, disp_c) = apply_synthetic_deformation(
            fixed, rotation_deg=5.0, translation=(12, -8),
            elastic_amplitude=3.0, gamma=1.08)

        methods = ['rigid', 'affine', 'bspline', 'demons']
        reg_map = {}
        for m in methods:
            print(f"    Running {m}...")
            if m == 'rigid':
                r, meta = rigid_registration(fixed, moving)
            elif m == 'affine':
                r, meta = affine_registration(fixed, moving)
            elif m == 'bspline':
                best_r, best_meta = None, {}
                best_tre = float('inf')
                for gs in GRID_SPACINGS:
                    r2, m2 = bspline_registration(fixed, moving, grid_spacing=gs)
                    if r2 is not None:
                        tre2, _, _, _, _ = compute_target_registration_error(
                            fixed, moving, r2, None, disp_r, disp_c)
                        if tre2 < best_tre:
                            best_tre = tre2
                            best_r, best_meta = r2, m2
                r, meta = best_r, best_meta
            elif m == 'demons':
                r, meta = demons_registration(fixed, moving)
            reg_map[m] = r if r is not None else moving

        results = {}
        for m in methods:
            r = reg_map[m]
            tre_m, tre_s, tre_med, tre_max, _ = compute_target_registration_error(
                fixed, moving, r, None, disp_r, disp_c)
            results[m] = {
                'tre_mean_um': float(tre_m * scale),
                'tre_std_um': float(tre_s * scale),
                'tre_median_um': float(tre_med * scale),
                'tre_max_um': float(tre_max * scale),
                'tre_mean_px': float(tre_m),
                'tre_std_px': float(tre_s),
                'pixel_size_um': scale,
                'ncc': float(ncc(fixed, r))
            }
            print(f"      TRE: {tre_m*scale:.3f}+-{tre_s*scale:.3f} um "
                  f"({tre_m:.4f}+-{tre_s:.4f} px), NCC: {results[m]['ncc']:.4f}")

        figure_real_comparison(fixed, moving, reg_map, dataset_name,
                                os.path.join(FIGURES_DIR, f'{dataset_name}_comparison.png'))
        figure_ablation(fixed, reg_map, results,
                         os.path.join(FIGURES_DIR, f'{dataset_name}_ablation.png'))
        generate_table(results,
                        os.path.join(TABLES_DIR, f'{dataset_name}_ablation.tex'),
                        f'Registration Results on {dataset_name} Visium')
        all_results[dataset_name] = results

    return all_results


def run_statistics(results_real):
    """Compute Mann-Whitney U for B-spline vs others."""
    print("\n" + "=" * 60)
    print("STATISTICAL TESTS")
    print("=" * 60)
    from scipy.stats import mannwhitneyu

    for dataset, results in results_real.items():
        print(f"\n  {dataset}:")
        for m in ['rigid', 'affine', 'demons']:
            if m in results and 'bspline' in results:
                ncc_b = results['bspline']['ncc']
                ncc_m = results[m]['ncc']
                print(f"    B-spline vs {m}: NCC {ncc_b:.4f} vs {ncc_m:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Reproduce all manuscript results')
    parser.add_argument('--no-synthetic', action='store_true')
    parser.add_argument('--no-real', action='store_true')
    parser.add_argument('--grid', type=int, nargs='+', default=GRID_SPACINGS)
    args = parser.parse_args()

    if args.grid != GRID_SPACINGS:
        GRID_SPACINGS[:] = args.grid

    ensure_dirs()

    results = {}

    if not args.no_synthetic:
        results['synthetic'] = run_synthetic()

    if not args.no_real:
        results['real_visium'] = run_real()
        run_statistics(results.get('real_visium', {}))

    with open(os.path.join(OUTPUT_DIR, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nAll results saved to {OUTPUT_DIR}/")
    print("Done.")
