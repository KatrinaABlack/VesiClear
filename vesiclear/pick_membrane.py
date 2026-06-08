#!/home/ptclfocuser/software/home/software/anaconda3/envs/vesicle-picker-samhq/bin/python
# IMPORTANT INFO
# Tested in the vesicle-picker-samhq conda env, but should run in any env with
# vesicle-picker and all Python dependencies.
# Run this in the vesicle-picker workflow in place of generate_picks.py
#
# PLAN A - Template Hybrid (deterministic, SciPy):
# Keeps the existing template/cross-correlation evidence-gathering, but replaces the
# old "threshold -> delete -> per-leaflet spline" with a single robust weighted-spline
# fit of the displacement field. See membrane_refine_scipy/plan_A_template_hybrid.md.


# Imports
from vesicle_picker import (
    postprocess,
    helpers,
    external_import,
    external_export
)
import numpy as np
from cv2 import GaussianBlur
from tqdm import tqdm
import sys
import os
import multiprocessing
import logging

# Load helper functions
from helpers.detailed_logging import set_worker_id, log_queued, log_result, merge_logs, init_detailed_logging, log_run_command
from helpers.picking_parser import parse_args
from helpers.save_output import write_splines_npy, write_splines_image, write_evidence_image
from helpers.time_tracker import TimeTracker
from helpers.load_micrographs import load_micrographs

# Load refinement pipeline functions
from refinement_pipeline.fit_centerline import refine_vesicle

MAX_PROCESSES = 10

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger(__name__)


def process_preloaded_data(args_tuple) -> tuple[int, int, dict[str, float]]:
    """
    Refine the vesicle membranes in a single micrograph, saving the inner /
    intermembrane / outer leaflet coordinates, and returning status (0 for success),
    total points in the fitted leaflets, and task durations.

    Each worker executes this function.
    """
    # Unpack arguments
    micrograph_tuple, params = args_tuple
    micrograph_dict, image_data, masks, uid = micrograph_tuple

    # Set worker ID for logging
    set_worker_id(os.getpid())

    # Extract processing parameters
    downsample = params['downsample']
    psize = params['psize']

    # Extract IO parameters
    picks_dir = params['picks_dir']
    spline_dir = params['spline_dir']
    logs_dir = params['logs_dir']

    # Initialize local timing variables
    time_tracker = TimeTracker()

    try:
        logger.info(f"Processing micrograph {uid}")

        num_masks = len(masks)

        # Use the pre-loaded data instead of fetching again
        image_fullres = image_data
        image_blurred = GaussianBlur(image_fullres, (29, 29), 5, 5)

        # Generate mask contours, reversing downsampling
        masks_edges = [postprocess.find_contour(mask) for mask in masks]
        masks_edges = [edges["contours"][0].squeeze(1) * downsample
                       for edges in masks_edges]

        # Refine each vesicle: smooth baseline -> gather evidence -> robust centerline
        splines = []
        all_evidence = []
        total_evidence = 0
        vesicles_fit = 0
        for contour in masks_edges:
            # Skip vesicles with too few contour points to fit a spline
            if len(contour) < 4:
                continue

            # ERROR HANDLING: a single bad vesicle should not stop the micrograph
            try:
                time_tracker.start_task("refine_vesicle")
                before = len(splines)
                evidence = refine_vesicle(contour, image_blurred, splines, uid, params)
                time_tracker.stop_task("refine_vesicle")
                if len(splines) > before:
                    vesicles_fit += 1
                if evidence is not None:
                    total_evidence += len(evidence['delta'])
                    if picks_dir is not None:
                        all_evidence.append(evidence)
            except (ValueError, IndexError, OverflowError) as e:
                print(e)
                time_tracker.stop_running_tasks()
                continue

        # Save the per-sample evidence overlay (QC of the correlation step)
        if picks_dir is not None:
            write_evidence_image(image_blurred, all_evidence, picks_dir, uid, psize)

        # Save final leaflet coordinates as arrays and an overlay image
        if spline_dir is not None:
            write_splines_npy(splines, spline_dir, uid)
            write_splines_image(image_blurred, splines, spline_dir, uid)

        logger.info(f"Completed processing micrograph {uid}")

        # Log success
        total_particles = sum(len(spline) for spline in splines)
        log_result(uid, 'SUCCESS', masks_found=num_masks, corr_matches=total_evidence,
                   cleaned_matches=vesicles_fit, final_picks=total_particles,
                   splines=len(splines) // 3, failure_reason='', logs_dir=logs_dir)

        return 0, total_particles, time_tracker.get_runtimes()

    except Exception as e:
        logger.error(f"Error processing {uid}: {e}", exc_info=True)
        time_tracker.stop_running_tasks()
        log_result(uid, 'ERROR', failure_reason=str(e)[:100], logs_dir=logs_dir)
        return 1, 0, {}


# Main execution section
def main():
    args, shared_parameters, cryosparc_parameters, process_parameters = parse_args(sys.argv)

    # Initialize detailed logging
    init_detailed_logging(args)
    logger.info(f"Detailed in-file logging initialized")

    # Save run command and settings for reproducibility
    log_run_command(sys.argv, args, shared_parameters)
    logger.info(f"Run info saved to {args.logs_dir}/run_info.txt")

    # Load micrographs from Cryosparc for processing
    logger.info("Establishing connection to cryoSPARC")
    project, micrographs = load_micrographs(cryosparc_parameters)
    logger.info(f"Retrieved {len(micrographs)} micrographs from cryoSPARC job {cryosparc_parameters['csparc_input_JID']}")

    # Force start method for multiprocessing to 'spawn' for better compatibility
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        logger.warning("Could not set start method to 'spawn', using default")

    # Now process in parallel without needing cryoSPARC connection in worker processes
    num_processes = min(multiprocessing.cpu_count(), MAX_PROCESSES)
    logger.info(f"Starting parallel processing with {num_processes} processes")

    # Process micrographs in batches to prevent memory overload
    batch_size = 10  # Adjust based on your available memory
    results = []
    total_processed = 0

    total_micrographs = len(micrographs)

    outer_pbar = tqdm(total=total_micrographs, desc="Overall progress", position=0)

    # Create a single persistent multiprocessing pool
    with multiprocessing.Pool(processes=num_processes) as pool:
        for batch_start in range(0, total_micrographs, batch_size):
            batch_end = min(batch_start + batch_size, total_micrographs)
            batch = micrographs[batch_start:batch_end]
            batch_size_actual = batch_end - batch_start
            logger.info(f"Processing batch {batch_start//batch_size + 1}, micrographs {batch_start} to {batch_end-1}")

            outer_pbar.update(batch_size_actual)

            preloaded_batch = []
            for micrograph in batch:
                uid = micrograph['uid']
                path = micrograph['micrograph_blob/path']
                micrograph_psize = micrograph['psize_A'] if 'psize_A' in micrograph.dtype.names else float(shared_parameters['psize'])

                masks_path = shared_parameters["masks_dir"] / f"{uid}_vesicles_filtered.pkl"
                if not os.path.isfile(masks_path):
                    logger.info(f"Skipping {uid}: Mask file not found")
                    continue

                # RESTART CAPABILITY: Check if already processed
                if process_parameters['spline_dir'] is not None:
                    first_spline_path = process_parameters['spline_dir'] / f"{uid}_vesicle_0_inner.npy"
                    if first_spline_path.exists():
                        logger.info(f"Skipping {uid}: Already processed (spline files exist)")
                        continue

                # Load micrograph image
                try:
                    logger.info(f"Loading mask and image for {uid}")
                    masks = external_import.import_masks_from_disk(masks_path)
                    header, image = project.download_mrc(path)
                    image = image[0]
                    logger.info(f"Successfully loaded data for {uid}")
                except Exception as e:
                    logger.warning(f"Skipping {uid}: {e}")
                    continue

                micrograph_dict = {
                    'uid': uid,
                    'micrograph_blob/path': path,
                    'micrograph_blob/shape': image.shape,
                    'psize_A': micrograph_psize,
                    'exp_group_id': micrograph['exp_group_id'] if 'exp_group_id' in micrograph.dtype.names else 0
                }

                # Log that this micrograph is queued for processing
                log_queued(uid)

                preloaded_batch.append((micrograph_dict, image, masks, uid))

            # Skip batch if no valid micrographs
            if len(preloaded_batch) == 0:
                logger.warning(f"Batch {batch_start//batch_size + 1} has no valid micrographs to process")
                continue

            # Package input for worker processes
            process_input_data = [(data, process_parameters) for data in preloaded_batch]

            chunksize = max(1, len(process_input_data) // (num_processes * 2))
            logger.info(f"Using chunksize of {chunksize} for batch")

            try:
                batch_results = list(tqdm(
                    pool.imap_unordered(process_preloaded_data, process_input_data, chunksize=chunksize),
                    total=len(process_input_data),
                    desc=f"Processing batch {batch_start//batch_size + 1}",
                    leave=False
                ))
            except Exception as e:
                logger.error(f"Batch {batch_start//batch_size + 1} failed: {e}")
                continue

            # Append results and update timing
            results.extend(batch_results)
            total_processed += len(batch_results)
            logger.info(f"Completed batch {batch_start//batch_size + 1}, total processed: {total_processed}")

            # Free memory
            import gc
            gc.collect()

        # Aggregate timing and particle count information
        total_time_tracker = TimeTracker()
        total_particles = 0
        for result in results:
            status, particles_count, timing = result
            if status == 0:  # Process terminated successfully
                total_particles += particles_count
                total_time_tracker.add_runtimes(timing)

        logger.info("Finished processing all micrographs")
        logger.info(f"Total picks: {total_particles}")

        # Merge worker logs into final results file
        logger.info("Merging worker logs...")
        merge_logs()
        logger.info("Log merge complete")

        # Record timing statistics
        logger.info(total_time_tracker)


if __name__ == "__main__":
    main()
