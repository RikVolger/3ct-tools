"""Tools to quickly check and compare images, to see if tube settings are OK"""
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

_plot_kwargs = {
    "figsize": (5, 5),
    "layout": "constrained",
}


def plot_intensity(image, threshold=50, vmin=0, vmax=None, 
                   title="Image, red is clipping", colormap=mpl.colormaps["bone"]):
    """Display image intensity with clipping highlights.

    Uses a matplotlib colormap and marks values above vmax in red and below
    vmin in blue. If `vmax` is not provided it is set based on 14-bit detector
    range minus the supplied threshold.

    Args:
        image (numpy.ndarray): 2D image array to display.
        threshold (int, optional): Subtracted from the 14-bit max to compute
            a default vmax when `vmax` is None. Defaults to 50.
        vmin (float, optional): Minimum value for colormap normalization.
            Defaults to 0.
        vmax (float, optional): Maximum value for colormap normalization.
            If None, set to 2**14 - threshold. Defaults to None.
        title (str, optional): Figure title. Defaults to "Image, red is clipping".
        colormap (matplotlib.colors.Colormap, optional): Colormap to use.
            The function will call `set_over("r")` and `set_under("b")` on it.
            Defaults to matplotlib's "bone" colormap.

    Returns:
        matplotlib.figure.Figure: The created figure containing the image.
    """
    # Maximum value to be mapped. Detectors report 14-bit values.
    if vmax is None:
        vmax = 2**14 - threshold

    colormap.set_over("r")
    colormap.set_under("b")

    fig, ax = plt.subplots(1, 1, *_plot_kwargs)
    ax.imshow(image, cmap=colormap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_aspect('equal')

    return fig


def plot_vertical_profile(image, line_position=400, label=None, fig=None):
    """Plot a vertical intensity profile (column) from an image.

    Args:
        image (numpy.ndarray): 2D image array.
        line_position (int, optional): X coordinate (column index) to sample.
            Defaults to 400.
        label (str, optional): Line label for the plot legend. Defaults to None.
        fig (matplotlib.figure.Figure, optional): Existing figure to reuse.
            If provided, the first axes of this figure will be used. Defaults to None.

    Returns:
        matplotlib.figure.Figure: Figure containing the vertical profile plot.
    """
    if fig is None:
        fig, ax = plt.subplots(1, 1, *_plot_kwargs)
        ax.set_title(f"Vertical profile at x = {line_position}")
    else:
        ax = fig.get_axes()[0]
    ax.plot(image[:, line_position], label=label)

    return fig


def plot_horizontal_profile(image, line_position=500, label=None, fig=None):
    """Plot a horizontal intensity profile (row) from an image.

    Args:
        image (numpy.ndarray): 2D image array.
        line_position (int, optional): Y coordinate (row index) to sample.
            Defaults to 500.
        label (str, optional): Line label for the plot legend. Defaults to None.
        fig (matplotlib.figure.Figure, optional): Existing figure to reuse.
            If provided, the first axes of this figure will be used. Defaults to None.

    Returns:
        matplotlib.figure.Figure: Figure containing the horizontal profile plot.
    """
    if fig is None:
        fig, ax = plt.subplots(1, 1, *_plot_kwargs)
        ax.set_title(f"Horizontal profile at y = {line_position}")
    else:
        ax = fig.get_axes()[0]

    ax.plot(image[line_position, :], label=label)

    return fig


def plot_vertical_relative(image1, image2, line_position=400):
    """Compute elementwise ratio of two images and plot a vertical profile.

    The relative image is computed as image1 / image2 (elementwise). Division
    by zero or invalid values will produce numpy inf/nan entries and may emit
    runtime warnings.

    Args:
        image1 (numpy.ndarray): Numerator image.
        image2 (numpy.ndarray): Denominator image (same shape as image1).
        line_position (int, optional): Column index to sample for the vertical
            profile. Defaults to 400.

    Returns:
        matplotlib.figure.Figure: Figure with the vertical profile of the ratio.
    """
    # divide image 1 by image 2 and then plot
    rel_image = image1 / image2
    fig = plot_vertical_profile(rel_image, line_position, label="relative profile")

    return fig


def plot_horizontal_relative(image1, image2, line_position=500):
    """Compute elementwise ratio of two images and plot a horizontal profile.

    The relative image is computed as image1 / image2 (elementwise). Division
    by zero or invalid values will produce numpy inf/nan entries and may emit
    runtime warnings.

    Args:
        image1 (numpy.ndarray): Numerator image.
        image2 (numpy.ndarray): Denominator image (same shape as image1).
        line_position (int, optional): Row index to sample for the horizontal
            profile. Defaults to 500.

    Returns:
        matplotlib.figure.Figure: Figure with the horizontal profile of the ratio.
    """
    rel_image = image1 / image2
    fig = plot_horizontal_profile(rel_image, line_position)

    return fig


def plot_diff(image1: np.ndarray, image2: np.ndarray):
    """Compute difference between two images and display it.

    Computes image1 - image2 and displays the resulting difference image with a
    diverging colormap (default "PuOr"). The plot color limits are set
    symmetrically around zero using the maximum absolute deviation in the
    difference (vmin = -max_abs, vmax = max_abs). This ensures positive and
    negative deviations use the same scale.

    Args:
        image1 (numpy.ndarray): First image.
        image2 (numpy.ndarray): Second image to subtract from the first.

    Returns:
        matplotlib.figure.Figure: Figure displaying the difference image.
    """
    diff_image = image1 - image2
    max_deviation = max(abs(diff_image.min()), abs(diff_image.max()))
    vmin = -max_deviation
    vmax = max_deviation
    fig = plot_intensity(diff_image, vmin=vmin, vmax=vmax, colormap=mpl.colormaps["PuOr"])
    return fig
