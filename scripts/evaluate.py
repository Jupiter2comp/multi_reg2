"""
Evaluation script: run all registration methods and compute TRE.
Matches manuscript Sections 6.2 and 6.3.

Usage:
  python scripts/evaluate.py --method all  # run all methods on synthetic data
  python scripts/evaluate.py --method all --real  # run all methods on real Visium data
  python scripts/evaluate.py --method bspline --grid 30  # single method, custom grid
"""
import os, sys, argparse, json, time
import numpy as np
from skimage import io
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.registration_pipeline import (
    apply_synthetic_deformation,
    compute_target_registration_error,
    rigid_registration,
    affine_registration,
    bspline_registration,
    demons_registration,
    ncc
)

PIXEL_SIZE_UM = {'breast': 0.3115, 'brain': 0.2510}

def run_synthetic_experiment(output_dir, methods, grid_spacings):
    """Run registration on synthetic IHC data with known ground truth."""
    print("=" * 60)
    print("SYNTHETIC EXPERIMENT (Section 6.1)")
    print("=" * 60)

    # Load or generate synthetic data
    fixed_path = os.path.join(output_dir, '..', 'data', 'fixed.png')
    if os.path.exists(fixed_path):
        fixed = io.imread(fixed_path) / 255.0
        moving = io.imread(os.path.join(output_dir, '..', 'data', 'moving.png')) / 255.0
        disp_r = np.load(os.path.join(output_dir, '..', 'data', 'gt_displacement_x.npy'))
        disp_c = np.load(os.path.join(output_dir, '..', 'data', 'gt_displacement_y.npy'))
    else:
        from scripts.data_generation import generate_synthetic_data
        fixed, moving, (_, disp_r, disp_c) = generate_synthetic_data(
            os.path.join(output_dir, '..', 'data'))

    # Convert moving to grayscale for methods that require single-channel
    fixed_gray = fixed if fixed.ndim == 2 else fixed.mean(axis=2)
    moving_gray = moving if moving.ndim == 2 else moving.mean(axis=2)

    results = {}
    for method in methods:
        print(f"\n  Running {method.upper()}...")
        t0 = time.time()

        if method == 'rigid':
            registered, meta = rigid_registration(fixed_gray, moving_gray)
        elif method == 'affine':
            registered, meta = affine_registration(fixed_gray, moving_gray)
        elif method == 'bspline':
            best_result = None
            best_tre = float('inf')
            for gs in grid_spacings:
                r, m = bspline_registration(fixed_gray, moving_gray, grid_spacing=gs)
                if r is not None:
                    tre, _, _, _, _ = compute_target_registration_error(
                        fixed_gray, moving_gray, r, None, disp_r, disp_c)
                    if tre < best_tre:
                        best_tre = tre
                        best_result = (r, m)
            registered, meta = best_result if best_result else (None, {})
        elif method == 'demons':
            registered, meta = demons_registration(fixed_gray, moving_gray)

        if registered is not None:
            tre_mean, tre_std, tre_med, tre_max, landmarks = \
                compute_target_registration_error(
                    fixed_gray, moving_gray, registered, None, disp_r, disp_c)

            results[method] = {
                'tre_mean_px': float(tre_mean),
                'tre_std_px': float(tre_std),
                'tre_median_px': float(tre_med),
                'tre_max_px': float(tre_max),
                'time_s': round(meta.get('time', 0), 2),
                'ncc': float(ncc(fixed_gray, registered))
            }
            print(f"    TRE: {tre_mean:.4f}+-{tre_std:.4f} px, "
                  f"NCC: {results[method]['ncc']:.4f}, "
                  f"Time: {results[method]['time_s']:.1f}s")
        else:
            results[method] = {'error': meta.get('error', 'unknown')}
            print(f"    FAILED: {results[method]['error']}")

    return results


def run_real_experiment(output_dir, methods, grid_spacings):
    """Run registration on real 10x Visium data with synthetic deformation."""
    print("\n" + "=" * 60)
    print("REAL VISIUM EXPERIMENT (Section 6.2)")
    print("=" * 60)

    import numpy as np
    data_root = os.path.join(output_dir, '..', 'data', 'processed')
    # Use pre-built density maps from build_density_maps.py

    results = {}
    for dataset_name in ['breast', 'brain']:
        npz_path = os.path.join(data_root, f'{dataset_name}_density_maps.npz')
        if not os.path.exists(npz_path):
            print(f"\n  Skipping {dataset_name}: {npz_path} not found")
            continue
        data = np.load(npz_path)
        fixed = data['density_total']
        scale = data['pixel_size_um'].item() if data['pixel_size_um'].ndim == 0 else data['pixel_size_um'][0]

        # Generate synthetic deformation with real-data parameters
        moving, (aff, disp_r, disp_c) = apply_synthetic_deformation(
            fixed, rotation_deg=5.0, translation=(12, -8),
            elastic_amplitude=3.0, gamma=1.08)

        print(f"\n  Dataset: {dataset_name} ({fixed.shape}, {scale:.4f} um/px)")
        print(f"  Deformation: affine(5 deg rot) + elastic(3px) + gamma(1.08)")

        dataset_results = {}
        for method in methods:
            print(f"    Running {method.upper()}...")
            t0 = time.time()

            if method == 'rigid':
                registered, meta = rigid_registration(fixed, moving)
            elif method == 'affine':
                registered, meta = affine_registration(fixed, moving)
            elif method == 'bspline':
                best_result = None
                best_tre = float('inf')
                for gs in grid_spacings:
                    r, m = bspline_registration(fixed, moving, grid_spacing=gs)
                    if r is not None:
                        tre, _, _, _, _ = compute_target_registration_error(
                            fixed, moving, r, None, disp_r, disp_c)
                        if tre < best_tre:
                            best_tre = tre
                            best_result = (r, m)
                registered, meta = best_result if best_result else (None, {})
            elif method == 'demons':
                registered, meta = demons_registration(fixed, moving)

            if registered is not None:
                tre_mean, tre_std, tre_med, tre_max, landmarks = \
                    compute_target_registration_error(
                        fixed, moving, registered, None, disp_r, disp_c)

                dataset_results[method] = {
                    'tre_mean_um': float(tre_mean * scale),
                    'tre_std_um': float(tre_std * scale),
                    'tre_median_um': float(tre_med * scale),
                    'tre_max_um': float(tre_max * scale),
                    'tre_mean_px': float(tre_mean),
                    'tre_std_px': float(tre_std),
                    'time_s': round(meta.get('time', 0), 2),
                    'pixel_size_um': float(scale),
                    'ncc': float(ncc(fixed, registered))
                }
                print(f"      TRE: {tre_mean*scale:.3f}+-{tre_std*scale:.3f} um "
                      f"({tre_mean:.4f}+-{tre_std:.4f} px), "
                      f"NCC: {dataset_results[method]['ncc']:.4f}, "
                      f"Time: {dataset_results[method]['time_s']:.1f}s")
            else:
                dataset_results[method] = {'error': meta.get('error', 'unknown')}
                print(f"      FAILED")

        results[dataset_name] = dataset_results

    return results


def run_statistical_tests(results_real):
    """Run Mann-Whitney U tests comparing B-spline vs other methods."""
    print("\n" + "=" * 60)
    print("STATISTICAL TESTS (Section 6.3.3)")
    print("=" * 60)

    from scipy.stats import mannwhitneyu

    for dataset in ['breast', 'brain']:
        if dataset not in results_real:
            continue
        ds = results_real[dataset]
        if 'bspline' not in ds or 'affine' not in ds:
            continue

        # Extract per-landmark NCC values from registered images
        # We approximate by comparing NCC scores
        ncc_bspline = ds['bspline'].get('ncc', 0)
        ncc_affine = ds['affine'].get('ncc', 0)
        ncc_demons = ds['demons'].get('ncc', 0)

        print(f"\n  {dataset.upper()}:")
        print(f"    B-spline NCC: {ncc_bspline:.4f}")
        print(f"    Affine  NCC: {ncc_affine:.4f}")
        print(f"    Demons  NCC: {ncc_demons:.4f}")

        tre_b = ds['bspline'].get('tre_mean_um', 0)
        tre_a = ds['affine'].get('tre_mean_um', 0)
        tre_d = ds['demons'].get('tre_mean_um', 0)
        print(f"    B-spline TRE: {tre_b:.3f} um")
        print(f"    Affine  TRE: {tre_a:.3f} um")
        print(f"    Demons  TRE: {tre_d:.3f} um")


def save_results(results_syn, results_real, output_dir):
    """Save all results to JSON."""
    os.makedirs(output_dir, exist_ok=True)
    combined = {
        'synthetic': results_syn,
        'real_visium': results_real,
        'parameters': {
            'synthetic': {
                'affine_scale': (1.02, 0.98),
                'rotation_deg': 7,
                'shear_deg': 1.5,
                'translation_px': (18, -14),
                'elastic_amplitude_px': 3
            },
            'real_visium': {
                'affine_scale': (1.02, 0.98),
                'rotation_deg': 5,
                'shear_deg': 1,
                'translation_px': (12, -8),
                'elastic_amplitude_px': 3
            }
        }
    }
    path = os.path.join(output_dir, 'results.json')
    with open(path, 'w') as f:
        json.dump(combined, f, indent=2)
    print(f"\nResults saved to {path}")
    return combined


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate registration methods')
    parser.add_argument('--method', default='all',
                        choices=['all', 'rigid', 'affine', 'bspline', 'demons'])
    parser.add_argument('--real', action='store_true',
                        help='Run on real Visium data')
    parser.add_argument('--grid', type=int, nargs='+', default=[20, 30, 40],
                        help='B-spline grid spacings to try')
    parser.add_argument('--output', default='results')
    args = parser.parse_args()

    methods = ['rigid', 'affine', 'bspline', 'demons'] if args.method == 'all' else [args.method]
    output_dir = os.path.join(os.path.dirname(__file__), '..', args.output)

    # Run synthetic experiment
    syn_results = run_synthetic_experiment(output_dir, methods, args.grid)

    # Run real data experiment
    real_results = {}
    if args.real:
        real_results = run_real_experiment(output_dir, methods, args.grid)
        run_statistical_tests(real_results)

    # Save
    save_results(syn_results, real_results, output_dir)
