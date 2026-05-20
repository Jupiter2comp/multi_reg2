"""
Synthetic deformation generation for controlled validation.
Matches manuscript Section 6.1.

Generates a known synthetic deformation on the IHC test image
from skimage.data.immunohistochemistry() and saves:
  - data/fixed.png   (original IHC reference image)
  - data/moving.png  (synthetically deformed moving image)
  - data/gt_displacement_x.npy (ground-truth displacement field, x-component)
  - data/gt_displacement_y.npy (ground-truth displacement field, y-component)

The deformation consists of:
  - Affine: scale (1.02,0.98), rotation 7 deg, shear 1.5 deg, translation (18,-14)
  - Elastic: sinusoidal basis functions, amplitude 3 px
  - Intensity: gamma modulation 1.08
"""
import os, sys
import numpy as np
from skimage import data, io, color
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.registration_pipeline import apply_synthetic_deformation


def generate_synthetic_data(output_dir='data'):
    """Generate synthetic IHC data with known ground-truth deformation."""
    os.makedirs(output_dir, exist_ok=True)

    # Load the IHC test image (manuscript Section 6.1)
    img = data.immunohistochemistry()
    if img.max() > 1.0:
        img = img / 255.0

    print(f"IHC image shape: {img.shape}")

    # Apply synthetic deformation with manuscript-default parameters
    warped, (aff, disp_r, disp_c) = apply_synthetic_deformation(
        img, rotation_deg=7.0, translation=(18, -14),
        elastic_amplitude=3.0, gamma=1.08)

    # Save images
    io.imsave(os.path.join(output_dir, 'fixed.png'), (img * 255).astype(np.uint8))
    io.imsave(os.path.join(output_dir, 'moving.png'), (warped * 255).astype(np.uint8))
    print(f"Saved fixed.png ({img.shape}), moving.png ({warped.shape})")

    # Save ground-truth displacement fields
    np.save(os.path.join(output_dir, 'gt_displacement_x.npy'), disp_r)
    np.save(os.path.join(output_dir, 'gt_displacement_y.npy'), disp_c)
    print(f"Saved gt_displacement_x.npy, gt_displacement_y.npy ({disp_r.shape})")

    print("Synthetic data generation complete.")
    return img, warped, (aff, disp_r, disp_c)


if __name__ == '__main__':
    generate_synthetic_data(output_dir=os.path.join(os.path.dirname(__file__), '..', 'data'))
