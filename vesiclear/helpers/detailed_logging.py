"""
Optimized logging for parallel processing - no I/O contention.
Each worker writes to its own temp file, merged at the end.
"""

import csv
from pathlib import Path
from time import strftime

# Global variables
LOGS_DIR = None
QUEUED_LOG = None
WORKER_ID = None

def init_file_logging(logs_dir=Path(".")):
    """Initialize the logging directory and queued log."""
    global LOGS_DIR, QUEUED_LOG
    LOGS_DIR = logs_dir
    QUEUED_LOG = LOGS_DIR / "queued.csv"
    
    # Create temp directory for worker logs
    temp_dir = LOGS_DIR / "temp_worker_logs"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize queued log
    with open(QUEUED_LOG, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['UID'])

def set_worker_id(worker_id):
    """Set the worker ID for this process."""
    global WORKER_ID
    WORKER_ID = worker_id

def log_queued(uid):
    """Log that a micrograph was queued for processing."""
    global QUEUED_LOG
    with open(QUEUED_LOG, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([uid])

def log_result(uid, status, masks_found=0, corr_matches=0, cleaned_matches=0, final_picks=0, splines=0, failure_reason='', logs_dir=None):
    """Log processing result to worker-specific temp file."""
    global LOGS_DIR, WORKER_ID
    
    # Set LOGS_DIR if provided (for worker processes)
    if logs_dir is not None:
        LOGS_DIR = logs_dir
    
    if LOGS_DIR is None:
        raise RuntimeError("LOGS_DIR not set - init_log() must be called first or logs_dir must be provided")
    
    if WORKER_ID is None:
        # Fallback for main process
        WORKER_ID = "main"
    
    worker_log = LOGS_DIR / "temp_worker_logs" / f"worker_{WORKER_ID}.csv"
    
    # Check if file exists to determine if we need header
    write_header = not worker_log.exists()
    
    with open(worker_log, 'a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                'UID', 'Status', 'Masks_Found', 'Correlation_Matches',
                'After_Cleaning', 'Final_Picks',
                'Splines_Generated', 'Failure_Reason'
            ])
        writer.writerow([
            uid, status, masks_found, corr_matches, 
            cleaned_matches, final_picks, splines, failure_reason
        ])

def merge_logs():
    """Merge all worker logs into final results.csv."""
    global LOGS_DIR
    
    results_file = LOGS_DIR / "results.csv"
    temp_dir = LOGS_DIR / "temp_worker_logs"
    
    # Collect all worker logs
    worker_logs = sorted(temp_dir.glob("worker_*.csv"))
    
    if not worker_logs:
        print("Warning: No worker logs found to merge")
        return
    
    # Write merged results
    with open(results_file, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow([
            'UID', 'Status', 'Masks_Found', 'Correlation_Matches',
            'After_First_Clean', 'After_Second_Clean', 'Final_Picks',
            'Splines_Generated', 'Failure_Reason'
        ])
        
        for worker_log in worker_logs:
            with open(worker_log, 'r') as infile:
                reader = csv.reader(infile)
                next(reader)  # Skip header
                for row in reader:
                    writer.writerow(row)
    
    print(f"Merged {len(worker_logs)} worker logs into {results_file}")


def init_detailed_logging(args):
    """
    Startup code to begin detailed logging, including creating the logs 
    directory and calling the function to intialize logging files.
    """
    logs_dir = Path(args.logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)
    # This function call is separate for now, as I'd like to rewrite file-based logging to not use global variables
    init_file_logging(logs_dir)


def log_run_command(argv, args, shared_parameters):
    run_info_path = Path(args.logs_dir) / "run_info.txt"
    with open(run_info_path, 'w') as f:
        f.write(f"Command: {' '.join(argv)}\n")
        f.write(f"Timestamp: {strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Parameters file: {args.parameters}\n\n")
        f.write("Command-line arguments:\n")
        for arg, value in vars(args).items():
            f.write(f"  --{arg}: {value}\n")
        f.write("\nShared parameters:\n")
        for key, value in shared_parameters.items():
            if key != 'parameters_obj':  # Skip the config object itself
                f.write(f"  {key}: {value}\n")
