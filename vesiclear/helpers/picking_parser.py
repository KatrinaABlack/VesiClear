from argparse import ArgumentParser, Namespace
from vesicle_picker import helpers
from pathlib import Path
import numpy as np

def parse_args(args_list: list[str]) -> tuple[Namespace, dict, dict, dict]:
    """
    Parse the provided command line arguments. Returns the parsed arguments
    (Namespace object of all arguments), a shared parameters dictionary,
    a CryoSparc parameters dictionary, and a process parameters dictionary.
    """
    # Define the Argument parser and all accepted arguments
    parser = ArgumentParser(
        prog="pick_membrane_robust.py",
        description="Pick refined membrane coordinates via a robust weighted-spline centerline fit"
    )
    parser.add_argument(
        "parameters",
        type=str,
        help="Path to .ini file containing parameters for vesicle picking"
    )
    parser.add_argument(
        "--membrane_template",
        type=str,
        help="Path of .npz file containing membrane template information"
    )
    parser.add_argument(
        "--contour_spacing",
        type=float,
        default=50,
        help="Separation in A between sample points on vesicle contours"
    )
    parser.add_argument(
        "--hist_endpoints",
        type=int,
        default=90,
        help="Distance in A for histogram to extend from the membrane"
    )
    parser.add_argument(
        "--spline_density",
        type=int,
        default=20000,
        help="Number of points to sample from the fitted centerline of each vesicle"
    )
    parser.add_argument(
        "--smoothing",
        type=float,
        default=15,
        help="RMS tolerance in A the baseline may deviate from the raw SAM contour. "
             "Higher = smoother baseline (removes mask jaggedness). v2 change."
    )
    parser.add_argument(
        "--fit_lambda",
        type=float,
        default=1.0,
        help="Roughness penalty for the penalized (P-spline) displacement fit. "
             "Higher = smoother centerline / stronger gap bridging. v3 change."
    )
    parser.add_argument(
        "--refine_iters",
        type=int,
        default=3,
        help="Inner prior passes: re-select peaks near the current fit, then refit. "
             "0 reproduces a single global fit. v3 change."
    )
    parser.add_argument(
        "--outer_iters",
        type=int,
        default=2,
        help="Outer passes: re-box perpendicular to the refined centerline and repeat "
             "the inner refinement. 1 = v3 behavior (no re-boxing). v5 change."
    )
    parser.add_argument(
        "--edge_border_px",
        type=float,
        default=10,
        help="A vesicle contour within this many px of the micrograph frame is treated as "
             "clipped; it is truncated to the in-frame arc and fit as an OPEN spline. v6 change."
    )
    parser.add_argument(
        "--min_arc_points",
        type=int,
        default=8,
        help="Skip an edge-truncated vesicle if fewer than this many contour points remain "
             "in-frame. v6 change."
    )
    parser.add_argument(
        "--similarity_threshold",
        type=float,
        default=0.1,
        help="Drop floor: picks with correlation below this get weight 0 (excluded from "
             "the fit and the prior; the spline bridges the gap). v3 change."
    )
    parser.add_argument(
        "--prior_band_start",
        type=float,
        default=40.0,
        help="Half-width in A of the prior search window on the first refinement pass."
    )
    parser.add_argument(
        "--prior_band_end",
        type=float,
        default=15.0,
        help="Half-width in A of the prior search window on the final refinement pass."
    )
    parser.add_argument(
        "--irls_loss",
        type=str,
        default="huber",
        choices=["huber", "tukey"],
        help="Robust influence function for IRLS outlier rejection"
    )
    parser.add_argument(
        "--irls_iters",
        type=int,
        default=3,
        help="Number of iteratively reweighted least-squares passes"
    )
    parser.add_argument(
        "--gap_min_arc",
        type=float,
        default=175.0,
        help="v9: do not draw the leaflet through a run of weak picks whose bridged arc "
             "length exceeds this (A); emit open arcs instead."
    )
    parser.add_argument(
        "--gap_confidence",
        type=float,
        default=0.2,
        help="v9: pick confidence below this counts as unsupported when detecting gaps."
    )
    parser.add_argument(
        "--no_gap_clip",
        action="store_true",
        help="v9: disable gap clipping (bridge all gaps and emit closed leaflets, as v6)."
    )
    parser.add_argument(
        "--no_gate_window",
        action="store_true",
        help="Disable zero-weighting of correlation peaks that land on the search-window edge"
    )
    parser.add_argument(
        "--open_contours",
        action="store_true",
        help="Treat contours as open strips instead of closed vesicles"
    )
    parser.add_argument(
        "--picks_dir",
        type=str,
        default=None,
        help="Path to save image of per-sample evidence (optional QC)"
    )
    parser.add_argument(
        "--spline_dir",
        type=str,
        default=None,
        help="Path to save np arrays and overlay of final membrane leaflet coordinates"
    )
    parser.add_argument(
        "--logs_dir",
        type=str,
        default=".",
        help="Path to save detailed processing logs"
    )

    # Call parse_args, generate the namespace object
    args = parser.parse_args(args_list[1:])

    # Generate the parameters dictionaries
    shared_parameters, cryosparc_parameters, process_parameters = generate_parameters_dicts(args)

    return args, shared_parameters, cryosparc_parameters, process_parameters


def generate_parameters_dicts(args: Namespace) -> tuple[dict, dict, dict]:
    """
    Generate the shared parameters dictionary, the CryoSparc parameters
    dictionary, and the process parameters dictionary from the provided parsed
    arguments.
    """
    # Read the .ini parameter file
    params_from_file = helpers.read_config(args.parameters)

    # Read the membrane template file
    membrane_template_params = np.load(args.membrane_template)

    shared_parameters = generate_shared_parameters(args, params_from_file, membrane_template_params)
    cryosparc_parameters = generate_cryosparc_parameters(args, params_from_file, membrane_template_params)
    process_parameters = generate_process_parameters(args, params_from_file, membrane_template_params)

    return shared_parameters, cryosparc_parameters, process_parameters


def _refinement_parameters(args, params_from_file, membrane_template_params) -> dict:
    """Parameters shared by the shared- and process-parameter dictionaries."""
    return {
        "downsample": int(params_from_file.get("general", "downsample")),
        "psize": float(params_from_file.get("general", "psize")),
        "contour_spacing": args.contour_spacing,
        "hist_offset": args.hist_endpoints,
        "spline_density": args.spline_density,
        "smoothing": args.smoothing,
        "fit_lambda": args.fit_lambda,
        "refine_iters": args.refine_iters,
        "outer_iters": args.outer_iters,
        "edge_border_px": args.edge_border_px,
        "min_arc_points": args.min_arc_points,
        "similarity_threshold": args.similarity_threshold,
        "prior_band_start": args.prior_band_start,
        "prior_band_end": args.prior_band_end,
        "irls_loss": args.irls_loss,
        "irls_iters": args.irls_iters,
        "gap_clip": not args.no_gap_clip,
        "gap_min_arc": args.gap_min_arc,
        "gap_confidence": args.gap_confidence,
        "gate_window": not args.no_gate_window,
        "baseline_closed": not args.open_contours,
        "picks_dir": Path(args.picks_dir) if args.picks_dir else None,
        "spline_dir": Path(args.spline_dir) if args.spline_dir else None,
        "membrane_template": list(membrane_template_params["intensity"]),
        "first_peak_offset": membrane_template_params["first_peak"].item(),
        "intermembrane_offset": membrane_template_params["intermembrane"].item(),
        "second_peak_offset": membrane_template_params["second_peak"].item(),
    }


def generate_shared_parameters(args: Namespace, params_from_file, membrane_template_params) -> dict:
    shared_parameters = _refinement_parameters(args, params_from_file, membrane_template_params)
    shared_parameters["masks_dir"] = Path(params_from_file.get("input", "directory"))
    shared_parameters["parameters_obj"] = params_from_file
    return shared_parameters


def generate_cryosparc_parameters(args: Namespace, params_from_file, membrane_template_params) -> dict:
    cryosparc_parameters = {
        "csparc_input_login": params_from_file.get("csparc_input", "login"),
        "csparc_input_PID": params_from_file.get("csparc_input", "PID"),
        "csparc_input_JID": params_from_file.get("csparc_input", "JID"),
        "csparc_input_type": params_from_file.get("csparc_input", "type")
    }
    return cryosparc_parameters


def generate_process_parameters(args: Namespace, params_from_file, membrane_template_params) -> dict:
    process_parameters = _refinement_parameters(args, params_from_file, membrane_template_params)
    process_parameters["logs_dir"] = Path(args.logs_dir)
    return process_parameters
