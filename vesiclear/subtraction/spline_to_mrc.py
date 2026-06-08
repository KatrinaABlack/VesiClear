import numpy as np
import os
from argparse import ArgumentParser
from pathlib import Path
import mrcfile
from tqdm import tqdm
import multiprocessing as mp
from functools import partial



primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113]


def parse_args() -> tuple[str, str, int, int, str, bool, float]:
    """
    Parse arguments of the script
    """
    parser = ArgumentParser(
        prog="spline_to_mrc.py",
        description="Convert a set of spline npy files to a mrc file"
    )
    parser.add_argument(
        "spline_dir", 
        type=str,
        help="Path to directory containing splines, name formatted as {uid}_vesicle_{i}_{inner|outer|intermembrane}.npy. The uid files should have no leading 0s"
    )
    parser.add_argument(
        "uids_file",
        type=str,
        help="File containing UIDs of splines to save (one per line)"
    )
    parser.add_argument(
        "x_len",
        type=int,
        help="Length for dim1 of output mrcfile"
    )
    parser.add_argument(
        "y_len",
        type=int,
        help="Length for dim2 of output mrcfile"
    )
    parser.add_argument(
        "mrc_dir",
        type=str,
        help="Path to directory to save output mrc files. Save as {uid}_{binary|prime}_splines.mrc"
    )
    parser.add_argument(
        "--save_binary",
        type=bool,
        default=True,
        help="Save as boolean (0 for no spline, 1 for a spline). Otherwise assign each vesicle a prime number key and save product of all vesicles at each location"
    )
    parser.add_argument(
        "--dilation_radius",
        type=int,
        default=18,
        help="Radius (in pixels) to dilate splines when converting to membrane"
    )
    args = parser.parse_args()
    return Path(args.spline_dir), args.uids_file, args.x_len, args.y_len, Path(args.mrc_dir), args.save_binary, args.dilation_radius


def collect_spline_files(spline_dir: str, uids_file: str) -> dict[str, list[str]]:
    """
    Return a dictionary mapping uids (as ints) to a list of .npy files containing the splines for that vesicle
    """
    uids = []
    with open(uids_file, "r") as f_uids:
        for line in f_uids:
            uids.append(int(line.rstrip("\n")))
    
    # List directory once and build lookup efficiently
    all_spline_files = os.listdir(spline_dir)
    uids_to_spline_files = {}
    for uid in tqdm(uids, desc="Collecting spline files", unit="UID"):
        uids_to_spline_files[uid] = [file for file in all_spline_files if file.startswith(str(uid))]
    
    return uids_to_spline_files
    
    
def merge_splines_binary(spline_dir: Path, spline_files: list, x_len: int, y_len: int):
    """
    Read a list of .npy spline files and create a binary np array merging their contents.
    """
    merged_splines = np.zeros((x_len, y_len), dtype=np.uint8)
    for spline_file in spline_files:
        if spline_file.endswith("intermembrane.npy"):
            continue
        spline = np.load(spline_dir / spline_file, allow_pickle=True)
        for particle in spline:
            # Spline saved on transposed image
            if 0 <= particle[1] < x_len and 0 <= particle[0] < y_len:
                merged_splines[particle[1], particle[0]] = 1
    return merged_splines
    
        
def merge_splines_prime(spline_dir: str, spline_files: str, x_len: int, y_len: int):
    """
    Read a list of .npy spline files and create a np array merging their contents, using prime keying to distinguish vesicles.
    """
    merged_splines = np.ones((x_len, y_len))
    for spline_file in spline_files:
        if spline_file.endswith("intermembrane.npy"):
            continue
        spline = np.load(spline_dir / spline_file).T  # Spline saved on transposed image
        vesicle_id = int(spline_file.split("_")[2])
        for particle in spline:
            # Spline saved on transposed image
            assert (0 <= particle[1] < x_len and 0 <= particle[0] < y_len), f"Particle out of spline bounds: {particle}"
            assert vesicle_id < 20, f"More than 20 vesicles! Add more primes to the list"
            merged_splines[particle[1], particle[0]] *= primes[vesicle_id]
    return merged_splines


import cv2

def dilate_splines(spline, dilation_radius):
    """
    Dilate a np array containing one or more splines, to expand it to the membrane area
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*dilation_radius+1, 2*dilation_radius+1))
    return cv2.dilate(spline, kernel).astype(np.int8)


def process_single_uid(uid, spline_files, spline_dir, x_len, y_len, mrc_dir, save_binary, dilation_radius):
    """
    Process a single UID: merge splines, dilate, and write to MRC file.
    This function is called by multiprocessing workers.
    """
    try:
        if save_binary:
            merged_splines = merge_splines_binary(spline_dir, spline_files, x_len, y_len)
            write_setting = "binary"
        else:
            merged_splines = merge_splines_prime(spline_dir, spline_files, x_len, y_len)
            write_setting = "prime"
        
        dilated_splines = dilate_splines(merged_splines, dilation_radius)
        output_path = mrc_dir / f"{uid:021}_{write_setting}_dilated{dilation_radius}_splines.mrc"
        mrcfile.write(output_path, dilated_splines)
        return uid, True, None
    except Exception as e:
        return uid, False, str(e)


def _process_wrapper(args):
    """Wrapper function to unpack arguments for multiprocessing."""
    return process_single_uid(*args)


def main():
    spline_dir, uids_file, x_len, y_len, mrc_dir, save_binary, dilation_radius = parse_args()
    uids_to_spline_files = collect_spline_files(spline_dir, uids_file)
    
    # Prepare arguments for each UID
    # Use reasonable number of processes (not too many to avoid memory issues)
    num_processes = min(max(1, mp.cpu_count() - 1), 16)  # Cap at 16 processes
    print(f"Processing {len(uids_to_spline_files)} UIDs using {num_processes} parallel processes...")
    
    # Create list of arguments for each UID
    args_list = [
        (uid, uids_to_spline_files[uid], spline_dir, x_len, y_len, mrc_dir, save_binary, dilation_radius)
        for uid in uids_to_spline_files
    ]
    
    # Process in parallel with progress bar
    failed_uids = []
    with mp.Pool(processes=num_processes) as pool:
        # Use imap_unordered with proper wrapper function (no lambda)
        with tqdm(total=len(args_list), desc="Processing UIDs", unit="UID") as pbar:
            for result in pool.imap_unordered(_process_wrapper, args_list, chunksize=1):
                uid, success, error = result
                if not success:
                    failed_uids.append((uid, error))
                pbar.update(1)
    
    if failed_uids:
        print(f"\nWarning: {len(failed_uids)} UIDs failed to process:")
        for uid, error in failed_uids[:10]:  # Show first 10 errors
            print(f"  UID {uid}: {error}")
        if len(failed_uids) > 10:
            print(f"  ... and {len(failed_uids) - 10} more")
    else:
        print(f"\nSuccessfully processed all {len(uids_to_spline_files)} UIDs!")
    

if __name__ == "__main__":
    main()

