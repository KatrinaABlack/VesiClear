import numpy as np
from scipy.interpolate import splprep, splev

def smooth_baseline(contour, params):
    # Fit a smooth periodic spline to the (jagged) SAM mask contour and resample it
    # at even arc-length spacing. Displacements are later measured relative to this
    # baseline, so its clean, stable normals replace the noisy per-segment normals.
    psize = params['psize']
    contour_spacing = params['contour_spacing']
    is_closed = params.get('baseline_closed', True)
    smoothing = params.get('smoothing', 15.0)  # RMS tolerance (A) the curve may deviate

    contour = np.asarray(contour, dtype=float)

    # Remove consecutive duplicate points (splprep requires distinct points)
    keep = np.ones(len(contour), dtype=bool)
    keep[1:] = np.any(np.diff(contour, axis=0) != 0, axis=1)
    contour = contour[keep]
    if len(contour) < 4:
        return None

    # Fit a SMOOTHING periodic cubic spline through the contour. s>0 is essential: an
    # interpolating (s=0) fit would follow every jag of the SAM mask, and that
    # lumpiness would propagate straight into the centerline. s ~ m * dev^2 lets the
    # curve deviate ~`smoothing` A RMS from the raw mask, removing mask roughness.
    dev_px = smoothing / psize
    s = len(contour) * (dev_px ** 2)
    try:
        tck, _ = splprep([contour[:, 0], contour[:, 1]], per=1 if is_closed else 0, s=s, k=3)
    except Exception:
        return None

    # Estimate perimeter from a dense evaluation, then resample evenly by arc length
    dense = np.array(splev(np.linspace(0, 1, 2000), tck))  # (2, 2000)
    seg = np.sqrt(np.sum(np.diff(dense, axis=1) ** 2, axis=0))
    cumulative = np.concatenate(([0.0], np.cumsum(seg)))
    perimeter_px = cumulative[-1]
    if perimeter_px < 1e-6:
        return None
    n_samples = max(8, int(perimeter_px * psize / contour_spacing))

    # Invert the (approximate) arc-length parameterization to get even spacing
    targets = np.linspace(0, perimeter_px, n_samples, endpoint=not is_closed)
    u_dense = np.linspace(0, 1, 2000)
    u_even = np.interp(targets, cumulative, u_dense)
    bx, by = splev(u_even, tck)
    baseline = np.column_stack((bx, by))

    # Enforce a consistent winding so the convention normal = [-(dy), dx] points OUT
    # of the vesicle. SAM/cv2 contours may be wound either way; the centerline fit is
    # orientation-invariant but the leaflet offsets are not, so we normalize here.
    # Works for open arcs too: the arc's own centroid sits toward the vesicle centre, so
    # the outward-normal sign test still holds for a curved arc.
    centroid = baseline.mean(axis=0)
    tangents = np.diff(baseline, axis=0)
    outward_normals = np.column_stack((-tangents[:, 1], tangents[:, 0]))
    midpoints = (baseline[:-1] + baseline[1:]) / 2.0
    if np.sum(np.einsum('ij,ij->i', outward_normals, midpoints - centroid)) < 0:
        baseline = baseline[::-1]

    return baseline


def _longest_inframe_run(inframe):
    # Longest contiguous run of True in a circular boolean array; returns index array.
    n = len(inframe)
    if inframe.all():
        return np.arange(n)
    start = np.where(~inframe)[0][0]          # rotate to begin at an off-frame point
    rolled = np.roll(inframe, -start)
    best = (0, 0)
    i = 0
    while i < n:
        if rolled[i]:
            j = i
            while j < n and rolled[j]:
                j += 1
            if j - i > best[1] - best[0]:
                best = (i, j)
            i = j
        else:
            i += 1
    return (np.arange(best[0], best[1]) + start) % n


def truncate_at_edges(contour, image_shape, border_px, min_points):
    # Detect a vesicle clipped by the micrograph frame (SAM contour runs along the image
    # border) and return the in-frame arc + is_closed flag:
    #   - no border-touching points -> (contour, True)   [closed vesicle, unchanged]
    #   - border-touching points    -> (longest in-frame arc, False)  [open, truncated]
    #   - too little in-frame        -> (None, None)      [skip]
    contour = np.asarray(contour, dtype=float)
    H, W = image_shape[-2:]
    x, y = contour[:, 0], contour[:, 1]
    on_frame = (x <= border_px) | (x >= W - 1 - border_px) | (y <= border_px) | (y >= H - 1 - border_px)
    if not on_frame.any():
        return contour, True
    inframe = ~on_frame
    if inframe.sum() < min_points:
        return None, None
    return contour[_longest_inframe_run(inframe)], False
