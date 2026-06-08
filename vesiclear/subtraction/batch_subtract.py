#!/usr/bin/env python3
"""Batch processing script for the spectral v4 membrane subtraction.

This script:
* Calls ``membrane_subtraction.py`` (safe version).
* Uses ``tqdm`` for a progress bar.
* Does NOT add an output suffix.
* Matches masks by UID + suffix.
"""
import argparse
import os
import glob
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
try:
    from tqdm import tqdm
except ImportError:
    print("tqdm not found, falling back to simple print")
    def tqdm(iterable, total=None, **kwargs):
        return iterable

def process_file(args):
    """Run a single subtraction job via ``subprocess``.
    
    Returns (infile, True/False, message)
    """
    infile, maskfile, outfile, script_path, sigma, ramp, dilate, use_gpu, python_exec = args
    cmd = [
        python_exec, script_path,
        "--input_mrc", infile,
        "--mask_mrc", maskfile,
        "--output_mrc", outfile,
        "--sigma", str(sigma),
        "--ramp", str(ramp),
        "--dilate", str(dilate)
    ]
    if use_gpu:
        cmd.append("--gpu")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return (infile, True, "")
    except subprocess.CalledProcessError as e:
        return (infile, False, e.stderr)

def main():
    parser = argparse.ArgumentParser(description="Batch process micrographs with spectral v4 (safe) subtraction.")
    parser.add_argument("--mic-dir", "-i", required=True, help="Directory containing input micrographs (.mrc)")
    parser.add_argument("--mask-dir", "-m", required=True, help="Directory containing mask files")
    parser.add_argument("--out-dir", "-o", required=True, help="Directory where output micrographs will be written")
    parser.add_argument("--mask-suffix", default="_binary_dilated28_splines.mrc", help="Suffix to append to UID to find mask")
    
    # Get the directory where this batch script resides
    script_dir = os.path.dirname(os.path.realpath(__file__))
    default_worker = os.path.join(script_dir, "membrane_subtraction.py")
    
    parser.add_argument("--script", default=default_worker, help="Path to the v4 subtraction script")
    parser.add_argument("--jobs", "-j", type=int, default=1, help="Number of parallel jobs")
    parser.add_argument("--sigma", type=float, default=50.0, help="Sigma for local statistics")
    parser.add_argument("--ramp", type=float, default=10.0, help="Ramp width (pixels)")
    parser.add_argument("--dilate", type=float, default=0.0, help="Mask dilation in Angstroms")
    parser.add_argument("--gpu", action="store_true", help="Use GPU if available")
    args = parser.parse_args()

    # Gather micrographs
    all_files = glob.glob(os.path.join(args.mic_dir, "*.mrc"))
    micrographs = []
    
    for f in all_files:
        if "binary_dilated" in f: 
            continue
        micrographs.append(f)
    micrographs.sort()

    tasks = []
    python_exec = sys.executable
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Save command to text file
    with open(os.path.join(args.out_dir, "run_command.txt"), "w") as f:
        f.write(" ".join(sys.argv) + "\n")
        
    print(f"Found {len(micrographs)} micrographs.")

    for mic in micrographs:
        base_filename = os.path.basename(mic)
        if '_' in base_filename:
            uid = base_filename.split('_')[0]
        else:
            uid = os.path.splitext(base_filename)[0]
            
        mask_name = f"{uid}{args.mask_suffix}"
        mask = os.path.join(args.mask_dir, mask_name)
        
        if not os.path.exists(mask):
             # Silent skip or log elsewhere to avoid cluttering tqdm?
             # print(f"[SKIP] No mask for {uid}")
             continue
             
        out = os.path.join(args.out_dir, base_filename)
        tasks.append((mic, mask, out, args.script, args.sigma, args.ramp, args.dilate, args.gpu, python_exec))

    print(f"Starting processing of {len(tasks)} file pairs with {args.jobs} jobs...")
    
    success_count = 0
    error_count = 0
    errors = []

    if args.jobs > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as executor:
            # map doesn't work easily with tqdm for updates as they complete, so use as_completed or submit
            futures = [executor.submit(process_file, t) for t in tasks]
            
            for future in tqdm(as_completed(futures), total=len(tasks), unit="mic"):
                infile, success, msg = future.result()
                if success:
                    success_count += 1
                else:
                    error_count += 1
                    errors.append(f"{os.path.basename(infile)}: {msg}")
    else:
        for t in tqdm(tasks, unit="mic"):
            infile, success, msg = process_file(t)
            if success:
                success_count += 1
            else:
                error_count += 1
                errors.append(f"{os.path.basename(infile)}: {msg}")

    print(f"\nBatch processing complete.")
    print(f"Success: {success_count}")
    print(f"Errors:  {error_count}")
    
    if errors:
        print("\n--- Error Log ---")
        for e in errors[:20]:
            print(e)
        if len(errors) > 20:
            print(f"... and {len(errors)-20} more errors.")

if __name__ == "__main__":
    main()
