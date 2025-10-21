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

    if fig is None:
        fig, ax = plt.subplots(1, 1, *_plot_kwargs)
        ax.set_title(f"Vertical profile at x = {line_position}")
    else:
        ax = fig.get_axes()[0]
    ax.plot(image[:, line_position], label=label)

    return fig


def plot_horizontal_profile(image, line_position=500, label=None, fig=None):

    if fig is None:
        fig, ax = plt.subplots(1, 1, *_plot_kwargs)
        ax.set_title(f"Horizontal profile at y = {line_position}")
    else:
        ax = fig.get_axes()[0]

    ax.plot(image[line_position, :], label=label)

    return fig


def plot_vertical_relative(image1, image2, line_position=400):
    # divide image 1 by image 2 and then plot
    rel_image = image1 / image2
    fig = plot_vertical_profile(rel_image, line_position, label="relative profile")

    return fig


def plot_horizontal_relative(image1, image2, line_position=500):
    rel_image = image1 / image2
    fig = plot_horizontal_profile(rel_image, line_position)

    return fig


def plot_diff(image1: np.ndarray, image2: np.ndarray):
    diff_image = image1 - image2
    vmin = diff_image.min()
    vmax = diff_image.max()
    fig = plot_intensity(diff_image, vmin=vmin, vmax=vmax, colormap=mpl.colormaps["PuOr"])
    return fig
