#!/usr/bin/env python3
import sys
import argparse
import numpy as np
import mrcfile
from scipy.ndimage import gaussian_filter, binary_dilation, distance_transform_edt
from scipy import fftpack

def generate_spectral_noise_clipped(shape, reference_data, mask, cutoff_factor=0.005):
    """
    Generate spectral noise with Input Clipping and High-Pass Filter.
    """
    # 1. Clip Input (+/- 5 Sigma)
    ref = reference_data.copy()
    mean = ref.mean()
    std = ref.std()
    low = mean - 5*std
    high = mean + 5*std
    ref = np.clip(ref, low, high)
    
    # 2. Fill mask with background mean so hole doesn't affect spectrum
    bg_pixels = ref[~mask]
    if bg_pixels.size == 0:
        # Fallback if mask covers entire image
        bg_mean = mean
    else:
        bg_mean = bg_pixels.mean()
        
    filled = ref.copy()
    filled[mask] = bg_mean
    
    # 3. Power Spectrum (FFT)
    reference_fft = fftpack.fft2(filled)
    power_spectrum = np.abs(reference_fft)
    
    # 4. Phase Randomization
    white_noise = np.random.normal(0, 1, size=shape)
    white_fft = fftpack.fft2(white_noise)
    white_phase = np.exp(1j * np.angle(white_fft))
    
    colored_fft = power_spectrum * white_phase
    
    # 5. High-Pass Filter (Old/Gentle Cutoff = 0.005)
    rows, cols = shape
    y, x = np.ogrid[:rows, :cols]
    y_dist = np.minimum(y, rows - y)
    x_dist = np.minimum(x, cols - x)
    radius = np.sqrt(y_dist**2 + x_dist**2)
    
    cutoff = cutoff_factor * max(rows, cols)
    if cutoff == 0: cutoff = 1e-6
    
    hp_filter = 1.0 - np.exp(-(radius**2) / (2 * cutoff**2))
    
    colored_fft *= hp_filter
    
    # 6. Inverse FFT
    colored_noise = np.real(fftpack.ifft2(colored_fft))
    
    # Normalize with safety check
    cn_std = colored_noise.std()
    if cn_std < 1e-9:
        # Avoid division by zero if noise is flat
        return np.zeros(shape)
        
    colored_noise -= colored_noise.mean()
    colored_noise /= cn_std
    
    return colored_noise

def fast_membrane_subtraction(input_path, mask_path, output_path, sigma=50.0, ramp=10.0, dilate_angstroms=5.0, seed=42):
    np.random.seed(seed)
    
    # 1. Load Data
    with mrcfile.open(input_path) as mrc:
        data = mrc.data.astype(np.float32)
        pixel_size = mrc.voxel_size.x
    
    with mrcfile.open(mask_path) as mrc:
        mask = mrc.data > 0.1
        
    if mask.shape != data.shape and mask.shape == data.shape[::-1]:
        print(f"  Transposing mask from {mask.shape} to {data.shape}...")
        mask = mask.T
    elif mask.shape != data.shape:
        raise ValueError(f"Mask shape {mask.shape} does not match input shape {data.shape}!")
        
    print(f"Processing {input_path}")
    print(f"  Pixel size: {pixel_size:.2f} A/px")
    print(f"  Sigma: {sigma}, Ramp: {ramp}, Dilate: {dilate_angstroms}A")
        
    # 2. Dilate Mask
    dilate_pixels = int(dilate_angstroms / pixel_size)
    if dilate_pixels > 0:
        print(f"  Dilating mask by {dilate_angstroms}A ({dilate_pixels}px)...")
        mask_dilated = binary_dilation(mask, iterations=dilate_pixels)
    else:
        print(f"  Skipping dilation ({dilate_pixels}px)...")
        mask_dilated = mask
    
    # 3. Local Statistics (Structure)
    print(f"  Computing local stats (sigma={sigma})...")
    local_mean = gaussian_filter(data, sigma=sigma)
    diff_sq = (data - local_mean)**2
    local_std = np.sqrt(gaussian_filter(diff_sq, sigma=sigma))
    
    # 4. Clipped Spectral Noise (Texture)
    print("  Generating Clipped Spectral Noise...")
    noise = generate_spectral_noise_clipped(data.shape, data, mask_dilated, cutoff_factor=0.005)
    
    # 5. Synthesis
    synthetic = local_mean + (local_std * noise)
    
    # 6. Ramp & Replace
    dist_map = distance_transform_edt(mask_dilated)
    prob_map = np.clip(dist_map / ramp, 0.0, 1.0)
    should_replace = (mask_dilated) & (np.random.random(data.shape) < prob_map)
    
    result = data.copy()
    result[should_replace] = synthetic[should_replace]
    
    # 7. Save
    print(f"  Saving to {output_path}...")
    with mrcfile.new(output_path, overwrite=True) as mrc:
        mrc.set_data(result)
        mrc.voxel_size = pixel_size
        
    print("  Done.")

def main():
    parser = argparse.ArgumentParser(description="Subract membrane using Clipped Spectral Noise")
    parser.add_argument("--input_mrc", required=True)
    parser.add_argument("--mask_mrc", required=True)
    parser.add_argument("--output_mrc", required=True)
    parser.add_argument("--sigma", type=float, default=50.0)
    parser.add_argument("--ramp", type=float, default=10.0)
    parser.add_argument("--dilate", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    fast_membrane_subtraction(
        args.input_mrc,
        args.mask_mrc,
        args.output_mrc,
        sigma=args.sigma,
        ramp=args.ramp,
        dilate_angstroms=args.dilate,
        seed=args.seed
    )

if __name__ == "__main__":
    main()
