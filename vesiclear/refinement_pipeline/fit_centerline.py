import numpy as np
from scipy.interpolate import splev, splprep, BSpline

from refinement_pipeline.extract_baseline import smooth_baseline, truncate_at_edges
from refinement_pipeline.gather_evidence import gather_profiles, select_peaks


def robust_weights(residuals, base_weights, loss, c, scale_floor=1.0):
    # Recompute weights for one IRLS pass: scale residuals by a robust MAD estimate
    # (centred on the median, floored so spikes are still flagged when the bulk fits
    # perfectly) and down-weight outliers via a Huber or Tukey influence function.
    median = np.median(residuals)
    mad = np.median(np.abs(residuals - median))
    scale = max(1.4826 * mad, scale_floor)
    r = (residuals - median) / scale
    if loss == 'tukey':
        factor = np.where(np.abs(r) <= c, (1 - (r / c) ** 2) ** 2, 0.0)
    else:  # huber
        factor = np.where(np.abs(r) <= c, 1.0, c / np.maximum(np.abs(r), 1e-9))
    return base_weights * factor


def pspline_fit(t, delta, weight, is_closed, total_arc_A, lam, loss, iters):
    # Penalized B-spline (P-spline) fit of the displacement field delta(t):
    #   minimize  sum_i w_i (delta_i - f(t_i))^2  +  lam * ||D^2 c||^2
    # The roughness penalty - not the data - governs regions with no weight, so
    # weight-0 picks contribute NOTHING and gaps are bridged smoothly (never pulled by
    # a bad pick). The penalty also caps effective flexibility, so IRLS retains its
    # ability to reject confident-wrong survivors. Returns a callable spline on [0,1].
    k = 3
    c_tune = 4.685 if loss == 'tukey' else 1.345

    # Pad across the seam for closed contours (continuity at the wrap)
    if is_closed:
        pad = 0.2
        left = t < pad
        right = t > 1.0 - pad
        tt = np.concatenate([t[right] - 1.0, t, t[left] + 1.0])
        dd = np.concatenate([delta[right], delta, delta[left]])
        ww = np.concatenate([weight[right], weight, weight[left]])
    else:
        tt, dd, ww = t.copy(), delta.copy(), weight.copy()

    # Normalize weights to median 1 and clip so no single pick out-votes the majority
    positive = ww[ww > 0]
    if positive.size < 4:
        return None
    w_median = np.median(positive)
    ww = np.clip(ww / (w_median if w_median > 0 else 1.0), 0.0, 1.5)

    # Uniform clamped cubic knot vector; one basis per ~50 A of contour, capped.
    lo, hi = tt.min(), tt.max()
    n_basis = int(np.clip(round(total_arc_A / 50.0), 10, 80))
    n_interior = max(1, n_basis - k - 1)
    interior = np.linspace(lo, hi, n_interior + 2)[1:-1]
    knots = np.concatenate(([lo] * (k + 1), interior, [hi] * (k + 1)))
    n_coef = len(knots) - k - 1

    order = np.argsort(tt)
    tt, dd, ww = tt[order], dd[order], ww[order]
    tt = np.clip(tt, lo, hi - 1e-9)
    B = BSpline.design_matrix(tt, knots, k).toarray()  # (m, n_coef)

    # Second-difference roughness penalty, scaled by data density so lam is stable
    D = np.diff(np.eye(n_coef), n=2, axis=0)
    penalty = lam * len(tt) * (D.T @ D)

    weights = ww.copy()
    coef = None
    for _ in range(max(1, iters)):
        W = weights[:, None]
        A = B.T @ (W * B) + penalty
        rhs = B.T @ (weights * dd)
        try:
            coef = np.linalg.solve(A, rhs)
        except np.linalg.LinAlgError:
            coef = np.linalg.lstsq(A, rhs, rcond=None)[0]
        residuals = dd - B @ coef
        weights = robust_weights(residuals, ww, loss, c_tune)

    if coef is None:
        return None
    return BSpline(knots, coef, k, extrapolate=True)


def _runs_true(mask):
    # contiguous runs of True in a 1D bool array -> list of (start, end) inclusive
    out, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            out.append((i, j)); i = j + 1
        else:
            i += 1
    return out


def _unsupported_runs(support, is_closed):
    # maximal runs of unsupported (False) samples; circular merge across the seam if closed.
    n = len(support)
    uns = ~np.asarray(support, dtype=bool)
    if not uns.any():
        return []
    idx = np.where(uns)[0]
    runs, s, prev = [], idx[0], idx[0]
    for k in idx[1:]:
        if k == prev + 1:
            prev = k
        else:
            runs.append([s, prev]); s, prev = k, k
    runs.append([s, prev])
    if is_closed and len(runs) >= 2 and runs[0][0] == 0 and runs[-1][1] == n - 1:
        first = runs.pop(0); last = runs.pop(-1)
        runs.append([last[0], first[1] + n])  # wrap run; indices taken mod n
    return runs


def _append_leaflet(center, normal, inner_rel, outer_rel, psize, splines):
    inner = center + (inner_rel / psize) * normal
    outer = center + (outer_rel / psize) * normal
    splines.append(np.round(inner).astype(int))
    splines.append(np.round(center).astype(int))
    splines.append(np.round(outer).astype(int))


def leaflets_from_points(center_pts, params, splines, weight=None):
    # Fit a smooth output curve through the centre points and emit the three leaflet
    # arrays (inner, intermembrane, outer) using the template's peak offsets.
    # v9 GAP CLIP: if per-sample `weight` is given and gap_clip is on, drop the output spans
    # that bridge a GENUINE gap - a run of LOW-CONFIDENCE samples (weight < gap_confidence;
    # this includes dropped weight==0 picks AND kept-but-weak ones) whose bridged arc length
    # exceeds gap_min_arc. The global fit is unchanged on supported arcs; we just do not draw
    # the leaflet through poorly-supported regions, emitting open arc(s) instead.
    psize = params['psize']
    spline_density = params['spline_density']
    is_closed = params.get('baseline_closed', True)
    inner_rel = params['first_peak_offset'] - params['intermembrane_offset']
    outer_rel = params['second_peak_offset'] - params['intermembrane_offset']
    gap_clip = params.get('gap_clip', False)
    gap_min_arc = params.get('gap_min_arc', 175.0)
    gap_confidence = params.get('gap_confidence', 0.2)  # below this = unsupported for clipping

    try:
        unique = np.concatenate(([True], np.any(np.diff(center_pts, axis=0) != 0, axis=1)))
        cpts = center_pts[unique]
        if len(cpts) < 4:
            return False
        per = 1 if is_closed else 0
        tck_c, u_pts = splprep([cpts[:, 0], cpts[:, 1]], per=per, s=0, k=3)
        u_pts = np.asarray(u_pts)
    except Exception:
        return False

    u = np.linspace(0, 1, spline_density, endpoint=not is_closed)
    cx, cy = splev(u, tck_c)
    dx, dy = splev(u, tck_c, der=1)
    nx, ny = -dy, dx
    nmag = np.sqrt(nx ** 2 + ny ** 2)
    nmag[nmag < 1e-9] = 1.0
    nx, ny = nx / nmag, ny / nmag
    center = np.column_stack((cx, cy))
    normal = np.column_stack((nx, ny))

    # Decide which dense spans to keep (drop only genuine-gap bridges)
    keep = np.ones(len(u), dtype=bool)
    if gap_clip and weight is not None:
        thr = gap_confidence if gap_confidence > 0 else 1e-12
        sup_c = np.asarray(weight, dtype=float)[unique] >= thr
        m = len(cpts)
        for a, b in _unsupported_runs(sup_c, is_closed):
            seg_idx = [k % m for k in range(a - 1, b + 2)]  # last good .. gap .. first good
            arc = cpts[seg_idx]
            glen = float(np.sum(np.linalg.norm(np.diff(arc, axis=0), axis=1)) * psize)
            if glen < gap_min_arc:
                continue  # short dropout -> bridge as in v6
            ua, ub = u_pts[(a - 1) % m], u_pts[(b + 1) % m]  # u of the supported neighbours
            if ua <= ub:
                keep &= ~((u > ua) & (u < ub))
            else:
                keep &= ~((u > ua) | (u < ub))  # bridge wraps the 0/1 seam

    if keep.all():
        _append_leaflet(center, normal, inner_rel, outer_rel, psize, splines)
        return True

    segs = _runs_true(keep)
    # a kept arc spanning the 0/1 seam shows up as a first + last segment -> join them
    if is_closed and len(segs) >= 2 and keep[0] and keep[-1]:
        first = segs.pop(0); last = segs.pop(-1)
        joined = list(range(last[0], last[1] + 1)) + list(range(first[0], first[1] + 1))
        segs.append(joined)
    emitted = False
    for seg in segs:
        idx = np.array(seg) if isinstance(seg, list) else np.arange(seg[0], seg[1] + 1)
        if len(idx) < 10:
            continue  # skip tiny fragments
        _append_leaflet(center[idx], normal[idx], inner_rel, outer_rel, psize, splines)
        emitted = True
    return emitted


def fit_inner(profiles_dict, uid, params):
    # ONE box-set: pick peaks globally, fit, then 3x prior-constrained re-pick + refit.
    # Returns (center_pts, evidence) - the refined per-sample centre points (which become
    # the baseline for the next outer pass) and the final picks for QC. No leaflet write.
    psize = params['psize']
    is_closed = params.get('baseline_closed', True)
    lam = params.get('fit_lambda', 1.0)
    irls_loss = params.get('irls_loss', 'huber')
    irls_iters = params.get('irls_iters', 3)
    refine_iters = params.get('refine_iters', 3)
    band_start = params.get('prior_band_start', 40.0)
    band_end = params.get('prior_band_end', 15.0)

    midpoints = profiles_dict['midpoints']
    normals = profiles_dict['normals']
    arc = profiles_dict['arc']
    total = arc[-1] if arc[-1] > 0 else 1.0
    t = arc / total
    total_arc_A = total * psize

    delta, weight = select_peaks(profiles_dict, params, prior_delta=None, band=None)
    spline = pspline_fit(t, delta, weight, is_closed, total_arc_A, lam, irls_loss, irls_iters)
    if spline is None:
        print(f"Centerline fit failed for {uid}")
        return None, None

    if refine_iters > 0:
        for band in np.linspace(band_start, band_end, refine_iters):
            prior = spline(t)
            delta, weight = select_peaks(profiles_dict, params, prior_delta=prior, band=band)
            new_spline = pspline_fit(t, delta, weight, is_closed, total_arc_A, lam, irls_loss, irls_iters)
            if new_spline is not None:
                spline = new_spline

    center_pts = midpoints + (spline(t) / psize)[:, None] * normals
    evidence = {'midpoints': midpoints, 'normals': normals, 'delta': delta, 'weight': weight}
    return center_pts, evidence


def refine_vesicle(contour, image, splines, uid, params):
    # Full v6 refinement of one vesicle.
    # EDGE HANDLING (v6): if the SAM contour is clipped by the micrograph frame, truncate
    # to the in-frame arc and process it as an OPEN spline, so the membrane is not forced
    # to close across off-frame space.
    # OUTER LOOP (v5): smooth contour -> baseline, then repeatedly box + pick + inner-refine
    # and re-box perpendicular to the refined line. Emits leaflets from the final points.
    outer_iters = params.get('outer_iters', 2)
    edge_border_px = params.get('edge_border_px', 10)
    min_arc_points = params.get('min_arc_points', 8)

    arc, is_closed = truncate_at_edges(contour, image.shape, edge_border_px, min_arc_points)
    if arc is None:
        return None  # too little of this vesicle is in-frame

    # per-vesicle params so the closed/open choice propagates through the whole pipeline
    vparams = dict(params)
    vparams['baseline_closed'] = is_closed

    baseline = smooth_baseline(arc, vparams)
    if baseline is None:
        return None

    center_pts, evidence = None, None
    for _ in range(max(1, outer_iters)):
        profiles = gather_profiles(baseline, image, vparams)
        if profiles is None:
            break
        cp, ev = fit_inner(profiles, uid, vparams)
        if cp is None:
            break
        center_pts, evidence = cp, ev
        baseline = cp  # re-box perpendicular to the refined line on the next pass

    if center_pts is None:
        return None
    weight = evidence['weight'] if evidence is not None else None
    leaflets_from_points(center_pts, vparams, splines, weight=weight)
    return evidence
