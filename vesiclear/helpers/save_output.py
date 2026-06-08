import matplotlib.pyplot as plt
import numpy as np


def write_edges_image(image, edges, picks_dir, uid, highlight_size=4):
    """
    Write the micrograph image with edge locations highlighted to uid.png in picks_dir
    """
    # Copy the image to add highlights
    image_out = np.copy(image)
    # Highlight value is maximum intensity
    highlight_val = np.max(image_out)

    # Apply highlighting at edge positions
    for edge in edges:
        for particle_trio in edge:
            for particle in particle_trio:
                # Highlight a box around the particle location
                for i in range(-highlight_size, highlight_size + 1):
                    for j in range(-highlight_size, highlight_size + 1):
                        if 0 <= particle[1] + i < image_out.shape[0] and 0 <= particle[0] + j < image_out.shape[1]:
                            image_out[particle[1] + i, particle[0] + j] = highlight_val
    
    # Save the highlighted image
    plt.imsave(picks_dir / f"{uid}.png", image_out, cmap="gray")

def write_rectangle_image(image, rectangles, picks_dir, uid, highlight_size=4):
    """
    Write the micrograph image with rectangle locations highlighted to uid.png in picks_dir
    """
    # Copy the image to add highlights
    image_out = np.copy(image)
    # Highlight value is maximum intensity
    highlight_val = np.max(image_out)

    # Apply highlighting at rectangle positions
    for rectangle in rectangles:
        for particle in rectangle:
            # Highlight particle in rectangle
            if 0 <= particle[1] < image_out.shape[0] and 0 <= particle[0] < image_out.shape[1]:
                image_out[particle[1], particle[0]] = highlight_val
    
    # Save the highlighted image
    plt.imsave(picks_dir / f"{uid}.png", image_out, cmap="gray")


def write_splines_npy(splines, spline_dir, uid):
    """
    Write the splines to npy files in spline_dir
    """
    for i in range(len(splines)//3):
        np.save(spline_dir / f"{uid}_vesicle_{i}_inner.npy", splines[3 * i + 0])
        np.save(spline_dir / f"{uid}_vesicle_{i}_intermembrane.npy", splines[3 * i + 1])
        np.save(spline_dir / f"{uid}_vesicle_{i}_outer.npy", splines[3 * i + 2])


def write_splines_image(image, splines, spline_dir, uid, highlight_size=4):
    """
    Write the micrograph image with splines highlighted to uid.png in spline_dir
    """
    # Copy the image to add highlights
    image_out = np.copy(image)
    # Highlight value is maximum intensity
    highlight_val = np.max(image_out)

    # Apply highlighting at spline positions
    for spline in splines:
        for particle in spline:
            # Highlight a box around the particle location
            for i in range(-highlight_size, highlight_size + 1):
                for j in range(-highlight_size, highlight_size + 1):
                    if 0 <= particle[1] + i < image_out.shape[0] and 0 <= particle[0] + j < image_out.shape[1]:
                        image_out[particle[1] + i, particle[0] + j] = highlight_val
    
    # Save the highlighted image
    plt.imsave(spline_dir / f"{uid}.png", image_out, cmap="gray")


def write_evidence_image(image, evidence_list, picks_dir, uid, psize):
    """
    Write a QC overlay of the per-sample membrane evidence to uid.png in picks_dir:
    the micrograph with each detected membrane-centre point coloured by its
    correlation weight (bright = confident, dark = weak / zero-weighted). This shows
    the raw evidence that feeds the robust centerline fit, before any smoothing.
    """
    # Use the object-oriented Agg API (no pyplot global state) for multiprocessing safety
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    fig = Figure(figsize=(10, 10))
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    ax.imshow(image, cmap="gray")

    # Detected membrane centre for each sample: baseline midpoint + delta along the normal
    xs, ys, weights = [], [], []
    for evidence in evidence_list:
        centers = evidence['midpoints'] + (evidence['delta'] / psize)[:, None] * evidence['normals']
        xs.extend(centers[:, 0])
        ys.extend(centers[:, 1])
        weights.extend(evidence['weight'])

    if xs:
        scatter = ax.scatter(xs, ys, c=weights, cmap="viridis", s=6, vmin=0)
        fig.colorbar(scatter, ax=ax, label="correlation weight", shrink=0.8)
    ax.set_axis_off()
    fig.savefig(picks_dir / f"{uid}.png", dpi=150, bbox_inches="tight")
