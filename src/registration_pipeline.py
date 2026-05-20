"""
Core implementation of the hierarchical registration pipeline (Phases 0-3).
Matches the methodology described in manuscript Section 5.

All registration methods use SimpleITK with Mattes Mutual Information metric.
Supports: Rigid (Euler2D), Affine, B-Spline (configurable grid spacing),
          and SyN-Demons (DemonsRegistrationFilter) baselines.
"""
import os, sys, time, warnings
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import ttest_ind
from scipy.ndimage import map_coordinates
import SimpleITK as sitk
from skimage import transform, color, exposure

warnings.filterwarnings('ignore')

# ========== Metric Functions ==========

def ncc(a, b):
    a = a.ravel().astype(np.float64); b = b.ravel().astype(np.float64)
    a -= a.mean(); b -= b.mean()
    d = np.sqrt(np.dot(a,a)*np.dot(b,b))
    return 0.0 if d < 1e-10 else float(np.dot(a,b)/d)

def mutual_info(a, b, bins=50):
    a = np.clip(a*255,0,255).astype(np.uint8).ravel()
    b = np.clip(b*255,0,255).astype(np.uint8).ravel()
    h, _, _ = np.histogram2d(a, b, bins=bins)
    pxy = h/np.sum(h)+1e-12; px=pxy.sum(1); py=pxy.sum(0)
    return float(np.sum(pxy*np.log(pxy/(px[:,None]*py[None,:]))))

def ssim(a, b):
    from skimage.metrics import structural_similarity
    return float(structural_similarity(a, b, data_range=b.max()-b.min()))

def all_metrics(fixed, moving, registered):
    if registered.ndim == 3: registered = color.rgb2gray(registered)
    if fixed.ndim == 3: fixed = color.rgb2gray(fixed)
    if moving.ndim == 3: moving = color.rgb2gray(moving)
    return {
        'ncc_unreg': ncc(fixed, moving), 'ncc_reg': ncc(fixed, registered),
        'mi_unreg': mutual_info(fixed, moving), 'mi_reg': mutual_info(fixed, registered),
        'ssim': ssim(fixed, registered),
    }

# ========== Registration Methods ==========

def rigid_registration(fixed, moving):
    """Phase 2: Coarse global rigid alignment using Euler2DTransform."""
    fi = sitk.GetImageFromArray(fixed.astype(np.float32))
    mi = sitk.GetImageFromArray(moving.astype(np.float32))
    try:
        init = sitk.CenteredTransformInitializer(
            fi, mi, sitk.Euler2DTransform(),
            sitk.CenteredTransformInitializerFilter.GEOMETRY)
    except RuntimeError:
        init = sitk.CenteredTransformInitializer(
            fi, mi, sitk.Euler2DTransform(),
            sitk.CenteredTransformInitializerFilter.MOMENTS)
    r = sitk.ImageRegistrationMethod()
    r.SetMetricAsMattesMutualInformation(64)
    r.SetMetricSamplingStrategy(r.RANDOM); r.SetMetricSamplingPercentage(0.25)
    r.SetInterpolator(sitk.sitkLinear)
    r.SetOptimizerAsGradientDescentLineSearch(1.0, 100, 1e-6, 5)
    r.SetOptimizerScalesFromPhysicalShift()
    r.SetInitialTransform(init, False)
    try:
        t0 = time.time()
        tf = r.Execute(fi, mi)
        elapsed = time.time() - t0
        res = sitk.Resample(mi, fi, tf, sitk.sitkLinear, 0)
        return sitk.GetArrayFromImage(res), {'time': elapsed, 'iterations': r.GetOptimizerIteration()}
    except RuntimeError as e:
        return None, {'error': str(e)}

def affine_registration(fixed, moving):
    """Affine registration (used as baseline)."""
    fi = sitk.GetImageFromArray(fixed.astype(np.float32))
    mi = sitk.GetImageFromArray(moving.astype(np.float32))
    try:
        init = sitk.CenteredTransformInitializer(
            fi, mi, sitk.Euler2DTransform(),
            sitk.CenteredTransformInitializerFilter.GEOMETRY)
    except RuntimeError:
        init = sitk.CenteredTransformInitializer(
            fi, mi, sitk.Euler2DTransform(),
            sitk.CenteredTransformInitializerFilter.MOMENTS)
    aff = sitk.AffineTransform(2)
    aff.SetCenter(init.GetCenter()); aff.SetMatrix(init.GetMatrix()); aff.SetTranslation(init.GetTranslation())
    r = sitk.ImageRegistrationMethod()
    r.SetMetricAsMattesMutualInformation(64)
    r.SetMetricSamplingStrategy(r.RANDOM); r.SetMetricSamplingPercentage(0.25)
    r.SetInterpolator(sitk.sitkLinear)
    r.SetOptimizerAsGradientDescentLineSearch(0.5, 100, 1e-6, 5)
    r.SetOptimizerScalesFromPhysicalShift()
    r.SetInitialTransform(aff, False)
    try:
        t0 = time.time()
        tf = r.Execute(fi, mi)
        elapsed = time.time() - t0
        res = sitk.Resample(mi, fi, tf, sitk.sitkLinear, 0)
        return sitk.GetArrayFromImage(res), {'time': elapsed, 'iterations': r.GetOptimizerIteration()}
    except RuntimeError as e:
        return None, {'error': str(e)}

def bspline_registration(fixed, moving, grid_spacing=20):
    """
    Phase 3: Fine local B-spline deformable registration.
    Rigid pre-alignment followed by B-spline refinement.
    Grid spacing of 20-40 px (manuscript Section 5.4, sensitivity Section 6.3.2).
    """
    fi = sitk.GetImageFromArray(fixed.astype(np.float32))
    mi = sitk.GetImageFromArray(moving.astype(np.float32))

    # Rigid pre-alignment
    try:
        init_r = sitk.CenteredTransformInitializer(
            fi, mi, sitk.Euler2DTransform(),
            sitk.CenteredTransformInitializerFilter.GEOMETRY)
    except RuntimeError:
        init_r = sitk.CenteredTransformInitializer(
            fi, mi, sitk.Euler2DTransform(),
            sitk.CenteredTransformInitializerFilter.MOMENTS)
    rr = sitk.ImageRegistrationMethod()
    rr.SetMetricAsMattesMutualInformation(64)
    rr.SetMetricSamplingStrategy(rr.RANDOM); rr.SetMetricSamplingPercentage(0.25)
    rr.SetInterpolator(sitk.sitkLinear)
    rr.SetOptimizerAsGradientDescentLineSearch(1.0, 30, 1e-5, 3)
    rr.SetOptimizerScalesFromPhysicalShift()
    rr.SetInitialTransform(init_r, False)
    try:
        tf_r = rr.Execute(fi, mi)
    except RuntimeError:
        tf_r = init_r

    # Warp moving into rigid space
    m_warped = sitk.Resample(mi, fi, tf_r, sitk.sitkLinear, 0)

    # B-spline refinement
    h, w = fixed.shape
    mx = min(max(4, int(w / grid_spacing)), 40)
    my = min(max(4, int(h / grid_spacing)), 30)
    bs = sitk.BSplineTransformInitializer(fi, [mx, my], 3)
    bs.SetIdentity()

    rb = sitk.ImageRegistrationMethod()
    rb.SetMetricAsMattesMutualInformation(64)
    rb.SetMetricSamplingStrategy(rb.RANDOM); rb.SetMetricSamplingPercentage(0.25)
    rb.SetInterpolator(sitk.sitkLinear)
    rb.SetOptimizerAsGradientDescentLineSearch(0.5, 50, 1e-4, 3)
    rb.SetOptimizerScalesFromPhysicalShift()
    rb.SetInitialTransform(bs, False)
    try:
        t0 = time.time()
        tf_b = rb.Execute(fi, m_warped)
        elapsed = time.time() - t0
        comp = sitk.CompositeTransform([tf_r, tf_b])
        res = sitk.Resample(mi, fi, comp, sitk.sitkLinear, 0)
        return sitk.GetArrayFromImage(res), {
            'time': elapsed, 'iterations': rb.GetOptimizerIteration(),
            'grid': [mx, my], 'grid_spacing': grid_spacing}
    except Exception as e:
        return None, {'error': str(e)}

def syn_demons_registration(fixed, moving):
    """SyN-like deformable baseline using DemonsRegistrationFilter."""
    fi = sitk.GetImageFromArray(fixed.astype(np.float32))
    mi = sitk.GetImageFromArray(moving.astype(np.float32))
    try:
        t0 = time.time()
        demons = sitk.DemonsRegistrationFilter()
        demons.SetNumberOfIterations(30)
        demons.SetSmoothDisplacementField(True)
        demons.SetStandardDeviations(1.0)
        disp = demons.Execute(fi, mi)
        tf = sitk.DisplacementFieldTransform(disp)
        elapsed = time.time() - t0
        res = sitk.Resample(mi, fi, tf, sitk.sitkLinear, 0)
        return sitk.GetArrayFromImage(res), {'time': elapsed}
    except Exception as e:
        return None, {'error': str(e)}

# ========== Synthetic Deformation Generation ==========
# Matches manuscript Section 6.1 parameters and Section 6.2 controlled-deformation setup.
# Synthetic: affine scale (1.02,0.98), rotation 7 deg, shear 1.5 deg, translation (18,-14)
# Real-data experiments: affine scale (1.02,0.98), rotation 5 deg, shear 1 deg, translation (12,-8)
# Both use sinusoidal elastic deformation (amplitude 3 px) and gamma modulation (1.08).

def apply_synthetic_deformation(img, rotation_deg=7.0, translation=(18, -14),
                                 elastic_amplitude=3.0, gamma=1.08):
    """
    Apply a known synthetic deformation to create a moving image.
    Returns (warped_image, deformation_params) where deformation_params
    contains the affine transform and elastic displacement field.
    """
    h, w = img.shape[:2]
    aff = transform.AffineTransform(
        scale=(1.02, 0.98), rotation=np.deg2rad(rotation_deg),
        shear=np.deg2rad(1.5), translation=translation)

    if img.ndim == 3:
        warped = transform.warp(img, aff.inverse, output_shape=(h, w), preserve_range=True)
    else:
        warped = transform.warp(img, aff.inverse, output_shape=(h, w))

    rows, cols = np.mgrid[0:h, 0:w].astype(np.float64)
    disp_r = elastic_amplitude * np.sin(cols / 30.0) + 0.5 * elastic_amplitude * np.sin((rows + cols) / 50.0)
    disp_c = elastic_amplitude * np.cos(rows / 40.0) + 0.33 * elastic_amplitude * np.sin(cols / 60.0)

    map_r = np.clip(rows + disp_r, 0, h-1)
    map_c = np.clip(cols + disp_c, 0, w-1)
    if img.ndim == 3:
        warped_el = np.zeros_like(warped)
        for c in range(3):
            warped_el[:,:,c] = map_coordinates(warped[:,:,c], [map_r, map_c], order=1, mode='reflect')
    else:
        warped_el = map_coordinates(warped, [map_r, map_c], order=1, mode='reflect')

    warped_el = exposure.adjust_gamma(warped_el, gamma)
    return np.clip(warped_el, 0, 1).astype(np.float32), (aff, disp_r, disp_c)

# ========== TRE Computation ==========
# Patch-based NCC matching against known ground-truth displacement field.
# Matches manuscript Section 6.2 methodology.

def compute_target_registration_error(fixed, warped, registered, aff, disp_r, disp_c, n_landmarks=80):
    """
    Compute TRE in pixels vs. known deformation ground truth.
    Uses NCC patch matching (17x17 patches) with search radius 12 px.
    """
    h, w = fixed.shape[:2]
    pts = np.mgrid[20:h-20:30, 20:w-20:30].reshape(2, -1).T
    if len(pts) == 0:
        return 0.0, 0.0, 0.0

    dr_interp = RegularGridInterpolator((np.arange(h), np.arange(w)), disp_r)
    dc_interp = RegularGridInterpolator((np.arange(h), np.arange(w)), disp_c)

    errors_px = []
    reg_gray = color.rgb2gray(registered) if registered.ndim == 3 else registered
    fixed_gray = color.rgb2gray(fixed) if fixed.ndim == 3 else fixed

    n = min(n_landmarks, len(pts))
    rng = np.random.RandomState(42)
    indices = rng.choice(len(pts), n, replace=False)

    for idx in indices:
        pr, pc = int(pts[idx, 0]), int(pts[idx, 1])
        if pr < 10 or pr >= h-10 or pc < 10 or pc >= w-10:
            continue
        patch = fixed_gray[pr-8:pr+9, pc-8:pc+9]
        if patch.shape != (17, 17):
            continue
        best_corr, best_dr, best_dc = -1.0, 0, 0
        for dr in range(-12, 13):
            for dc in range(-12, 13):
                sr, sc = pr+dr, pc+dc
                if sr-8 < 0 or sr+9 > h or sc-8 < 0 or sc+9 > w:
                    continue
                corr = ncc(patch, reg_gray[sr-8:sr+9, sc-8:sc+9])
                if corr > best_corr:
                    best_corr = corr
                    best_dr, best_dc = dr, dc
        gt_r = float(dr_interp(np.array([[pr, pc]])))
        gt_c = float(dc_interp(np.array([[pr, pc]])))
        error = np.sqrt((best_dr - gt_r)**2 + (best_dc - gt_c)**2)
        errors_px.append(error)

    if not errors_px:
        return 0.0, 0.0, 0.0
    err = np.array(errors_px)
    return float(np.mean(err)), float(np.std(err)), float(np.median(err))

# ========== Pipeline Orchestrator ==========
# Matches Algorithm 1 in manuscript Section 5.

def run_full_pipeline(fixed_image, moving_image, pixel_size_um=1.0,
                      do_rigid=True, do_affine=True, do_bspline=True,
                      bspline_grid_spacings=(20, 30, 40), do_syn=True,
                      verbose=True):
    """
    Run the full hierarchical pipeline plus all baselines.
    Returns a dict of results keyed by method name.
    """
    fixed_gray = color.rgb2gray(fixed_image).astype(np.float32) if fixed_image.ndim == 3 else fixed_image.astype(np.float32)
    moving_gray = color.rgb2gray(moving_image).astype(np.float32) if moving_image.ndim == 3 else moving_image.astype(np.float32)

    # Normalize
    for arr in [fixed_gray, moving_gray]:
        arr -= arr.min()
        arr /= (arr.max() + 1e-10)

    methods = {}
    methods['Unregistered'] = lambda: (moving_gray.copy(), {'time': 0.0})
    if do_rigid:
        methods['Rigid'] = lambda: rigid_registration(fixed_gray, moving_gray)
    if do_affine:
        methods['Affine'] = lambda: affine_registration(fixed_gray, moving_gray)
    if do_bspline:
        for gs in bspline_grid_spacings:
            methods[f'BSpline_GS{gs}'] = lambda gs=gs: bspline_registration(fixed_gray, moving_gray, grid_spacing=gs)
    if do_syn:
        methods['SyN_Demons'] = lambda: syn_demons_registration(fixed_gray, moving_gray)

    results = {}
    for name, func in methods.items():
        if verbose:
            print(f"  Running {name}...", end=' ', flush=True)
        t0 = time.time()
        try:
            reg, info = func()
            if reg is None:
                if verbose:
                    print(f"FAILED: {info.get('error', '?')}")
                results[name] = {'error': info.get('error', 'Failed')}
                continue
        except Exception as e:
            import traceback
            if verbose:
                print(f"EXCEPTION: {e}")
            results[name] = {'error': str(e)}
            continue

        if reg.ndim == 3:
            reg_gray = color.rgb2gray(reg)
        else:
            reg_gray = reg

        if reg_gray.shape != fixed_gray.shape:
            import cv2
            reg_gray = cv2.resize(reg_gray, (fixed_gray.shape[1], fixed_gray.shape[0]))

        met = all_metrics(fixed_gray, moving_gray, reg_gray)
        results[name] = {
            'registered': reg_gray,
            'ncc': met['ncc_reg'],
            'mi': met['mi_reg'],
            'ssim': met['ssim'],
            'time': info.get('time', 0.0),
            'info': info,
        }
        if verbose:
            print(f"NCC={met['ncc_reg']:.4f} MI={met['mi_reg']:.4f} Time={info.get('time',0):.1f}s")

    return results

# ========== Statistical Tests ==========

def compute_statistics(results):
    """t-test comparing B-spline vs. other methods. Matches Section 6.2."""
    bs_ncc = [v['ncc'] for k, v in results.items() if 'BSpline' in k and 'ncc' in v]
    ot_ncc = [v['ncc'] for k, v in results.items()
              if 'BSpline' not in k and 'ncc' in v and k != 'Unregistered']
    if bs_ncc and ot_ncc:
        t, p = ttest_ind(bs_ncc, ot_ncc)
        return {'t_statistic': float(t), 'p_value': float(p),
                'bspline_mean_ncc': float(np.mean(bs_ncc)),
                'others_mean_ncc': float(np.mean(ot_ncc))}
    return None
