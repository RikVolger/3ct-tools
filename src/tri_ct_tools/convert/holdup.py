import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

_plot_kwargs = {
    "figsize": (5, 5),
    "layout": "constrained",
}


def two_point_holdup(meas, full, empty):
    """Calculate holdup fraction from measurement and reference images.

    Computes the gas fraction in a two-phase system using the two-point method:
    holdup = ln(meas/full) / ln(empty/full). Values are clipped to [0, 1].

    Args:
        meas (np.ndarray): Measurement image(s) with two-phase mixture.
        full (np.ndarray): Reference image with liquid phase only (same shape as meas).
        empty (np.ndarray): Reference image with gas phase only (same shape as meas).

    Returns:
        np.ndarray: Holdup fraction array with values in [0, 1] (same shape as input).
    """
    holdup = np.log(meas / full) / np.log(empty / full)
    np.clip(holdup, 0, 1, out=holdup)
    return holdup


def plot_holdup(image, vmin=0, vmax=0.6, title="Holdup", colormap=mpl.colormaps["viridis"]):
    """Display a holdup image with highlights for values outside the range.

    The function shows `image` using the provided `colormap` and marks values
    above `vmax` as "over" (colored red) and values below `vmin` as "under"
    (colored blue) by calling `set_over("r")` and `set_under("b")` on the
    colormap object. The colormap is modified in-place.

    Args:
        image (numpy.ndarray): 2D array of holdup values to display.
        vmin (float, optional): Minimum value for colormap normalization.
            Values strictly below `vmin` will be shown with the colormap's
            under color (default: 0).
        vmax (float, optional): Maximum value for colormap normalization.
            Values strictly above `vmax` will be shown with the colormap's
            over color (default: 0.6).
        title (str, optional): Figure title to display above the image
            (default: "Holdup").
        colormap (matplotlib.colors.Colormap, optional): A matplotlib
            colormap instance used to render the image. The function will
            call `set_over("r")` and `set_under("b")` on this object, so
            note that the colormap is changed in-place. By default the
            function uses matplotlib's "viridis" colormap.

    Returns:
        tuple: (matplotlib.figure.Figure, matplotlib.image.AxesImage)
            The created figure and the AxesImage returned by ``imshow``.
    """

    colormap.set_over("r")
    colormap.set_under("b")

    fig, ax = plt.subplots(1, 1, **_plot_kwargs)
    im = ax.imshow(image, cmap=colormap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.axis('off')

    return fig, im
