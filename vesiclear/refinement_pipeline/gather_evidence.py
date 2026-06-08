import numpy as np
from refinement_pipeline.plot_intensity import rectangle_corners, points_in_rectangle, bin_rectangle_optimized


def gather_profiles(baseline, image, params):
    # Build the per-sample geometry and binned perpendicular intensity profiles ONCE.
    # The expensive rectangle rasterization lives here; peak selection (select_peaks)
    # is cheap and re-run each refinement iteration against the current prior.
    psize = params['psize']
    is_closed = params.get('baseline_closed', True)

    n = len(baseline)
    n_pairs = n if is_closed else n - 1

    midpoints, normals, arcs, profiles = [], [], [], []
    cumulative = 0.0
    for i in range(n_pairs):
        p1 = baseline[i]
        p2 = baseline[(i + 1) % n]
        seg_len = np.linalg.norm(p2 - p1)
        if seg_len < 1e-6:
            continue

        midpoint = (p1 + p2) / 2.0
        normal = np.array([-(p2[1] - p1[1]), (p2[0] - p1[0])], dtype=float)
        normal = normal / np.linalg.norm(normal)
        arc_mid = cumulative + seg_len / 2.0
        cumulative += seg_len

        # A single degenerate segment must skip only this sample, not the vesicle.
        try:
            corners = rectangle_corners(p1, p2, psize)
            rectangle = np.array(points_in_rectangle(*corners))
            intensities = bin_rectangle_optimized(image, p1, p2, rectangle, params)
        except Exception:
            continue

        midpoints.append(midpoint)
        normals.append(normal)
        arcs.append(arc_mid)
        profiles.append(intensities)

    if len(profiles) < 6:
        return None

    return {
        'midpoints': np.array(midpoints),
        'normals': np.array(normals),
        'arc': np.array(arcs),
        'profiles': np.array(profiles, dtype=float),  # (n_samples, 2*hist_offset+1)
    }


def select_peaks(profiles_dict, params, prior_delta=None, band=None):
    # Cross-correlate each cached profile with the template and choose the membrane
    # displacement. With a prior the search is restricted to +/-band (A) around the
    # predicted position, so a pick physically cannot drift to a far-off wrong feature.
    # Picks below the drop floor (similarity_threshold) get weight 0: they neither
    # influence the fit nor anchor the prior - the spline bridges those gaps instead.
    template = np.asarray(params['membrane_template'], dtype=float)
    hist_offset = params['hist_offset']
    center_offset = params['intermembrane_offset']
    gate_window = params.get('gate_window', True)
    drop = params.get('similarity_threshold', 0.1)

    profiles = profiles_dict['profiles']
    n = len(profiles)
    deltas = np.zeros(n)
    weights = np.zeros(n)

    for i in range(n):
        intensity = profiles[i]
        intensity_range = np.max(intensity) - np.min(intensity)
        if intensity_range < 1e-10:
            continue  # flat profile -> weight 0

        norm = (intensity - np.min(intensity)) / intensity_range
        norm = (norm - 0.5) * 2
        scores = np.correlate(norm, template) / template.shape[0]
        candidate_delta = np.arange(len(scores)) + center_offset - hist_offset

        if prior_delta is not None and band is not None:
            mask = np.abs(candidate_delta - prior_delta[i]) <= band
            if not mask.any():
                continue  # nothing near the prior -> weight 0, bridge
            masked = np.where(mask, scores, -np.inf)
            k = int(np.argmax(masked))
        else:
            k = int(np.argmax(scores))
            if gate_window and (k == 0 or k == len(scores) - 1):
                continue  # peak hit the search-window edge -> unreliable

        w = float(scores[k])
        if w < drop:
            continue  # below drop floor -> weight 0
        deltas[i] = candidate_delta[k]
        weights[i] = w

    return deltas, weights
