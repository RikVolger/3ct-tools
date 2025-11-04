import numpy as np
import matplotlib.pyplot as plt

from tri_ct_tools.convert.geometry import calc_distances


def single_cam_analysis(geoms_all_cams, images, cam, det, rows=[720, 780]):
    d, _ = calc_distances(geoms_all_cams, cam, det)

    image_full, image_empty = images

    row_start, row_end = rows

    col_start = 0
    col_end = -1
    # if cam == 0:
    #     col_start = 40
    #     col_end = 1450
    # elif cam == 1:
    #     col_start = 100
    #     col_end = 1520
    # elif cam == 2:
    #     col_start = 60
    #     col_end = 1480

    d = d[row_start:row_end, col_start:col_end]
    image_empty = image_empty[row_start:row_end, col_start:col_end]
    image_full = image_full[row_start:row_end, col_start:col_end]

    ln_intensity = -np.log(image_full / image_empty)

    image_overlay(cam, d, image_full, image_empty, row_start, col_start, ln_intensity)

    # distance through water
    d_flat = d.flatten()

    ln_intensity_flat = ln_intensity.flatten()
    effective_attenuation = ln_intensity_flat / d_flat

    intensity_attenuation_plot(cam, d_flat, ln_intensity_flat, effective_attenuation)

    # Define the value range
    min_value = 16000
    max_value = 20000

    # Find the locations where the values fall within the range
    points = np.where(
        (image_full >= min_value) &
        (image_full <= max_value))
    no_column = d_flat == 0

    plot_outliers(image_full, points, no_column)


def plot_outliers(image_full, points, no_column):
    fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].imshow(image_full)

    # Overlay red dots on the points where the values are between 2500 and 6000
    ax[0].scatter(points[1], points[0], color='red', marker='o', s=5)  # Red dots for values in range
    ax[0].scatter(no_column[1], no_column[0], color='blue', marker='o', s=5, alpha=0.1)
    ax[0].set_aspect('equal')

    ax[1].imshow(image_full)
    ax[1].set_aspect('equal')
    return fig


def intensity_attenuation_plot(cam, d, ln_intensity, effective_attenuation):
    mask = d > 0
    fig, ax = plt.subplots(1, 2, figsize=(15, 5), layout='tight')
    fig.suptitle(f"Detector {cam+1}, geom {cam+1}")
    ax[0].plot(d[mask], ln_intensity[mask], '--o', alpha=0.5)
    ax[0].set_xlabel('x (cm)', fontsize=15)
    ax[0].set_ylabel(R'$-ln(\frac{I(x)}{I_{empty}})$', fontsize=15)

    ax[1].plot(d[mask], effective_attenuation[mask], '--o', alpha=0.5)
    ax[1].set_xlabel('x (cm)', fontsize=15)
    ax[1].set_ylabel(R'$\mu_{eff}$'+' (1/cm)', fontsize=15)
    return fig


def image_overlay(cam, d, image_full, image_empty, row_start, col_start, ln_intensity):
    """Generates an image overlaying the full image with distance through the column

    The function creates a 3-channel image (RGB) where red represents the relative
    intensity in the full image, and green represents the relative distance through
    the column. Blue channel is unused as of now.

    Args:
        cam (int): Camera number
        d (np.ndarray): Distance through water for every pixel in the image
        image_full (np.ndarray): Image where the column is full of water
        image_empty (np.ndarray): Image without water in the column
        row_start (int): The bottom pixel of the region of interest
        col_start (int): The left pixel of the region of interest
        ln_intensity (np.ndarray): Logarithmic relative intensity between full and empty

    Returns:
        plt.figure.Figure: Figure containing the plot
    """
    multichannel_im = np.zeros((*image_full.shape, 3))
    multichannel_im[..., 0] = image_full / image_full.max()
    multichannel_im[..., 1] = d / d.max()
    # multichannel_im[..., 2] = d_outer / d_outer.max()

    # image_full_norm_inv = d.max() - (image_full / image_full.max() * d.max())
    image_full_inv = image_full.max() - image_full
    image_empty_inv = image_empty.max() - image_empty

    # rel_img = image_full / image_empty
    # rel_img_inv = rel_img.max() - rel_img
    # rel_img_inv_norm = rel_img / rel_img.max() * image_full_inv.max()
    # rel_img_norm = rel_img / rel_img.max() * d.max()

    hline = 750 - row_start
    # vline = 750 - col_start

    fig, ax = plt.subplots(1, 3, figsize=(15, 5), layout='constrained')
    ax[0].imshow(multichannel_im)
    ax[0].set_title(f"detector {cam+1}")

    # ax[1].plot(d_outer[hline, :], '.', label="d_outer")
    ax[1].plot(d[hline, :], '.', label="d_inner")
    ax1r = ax[1].twinx()
    ax1r.plot(ln_intensity[hline, :], label="$-log(I_{full}/I_{empty})$")
    # ax1r.plot(image_empty_inv[hline, :], label="Empty")
    ax[1].set_title(f"Horizontal line {hline}")
    ax[1].set_ylabel("cm")
    ax[1].legend(loc="center")
    ax1r.legend(loc="lower center")

    # ax[2].semilogy(d_outer[hline, :], '.', label="d_outer")
    ax[2].semilogy(d[hline, :], '.', label="d_inner")
    ax1r = ax[2].twinx()
    ax1r.semilogy(image_full_inv[hline, :], label="Full")
    ax1r.semilogy(image_empty_inv[hline, :], label="Empty")
    ax[2].set_title(f"Horizontal line {hline}")
    ax[2].set_ylabel("cm")
    ax[2].legend(loc="center")
    ax1r.legend(loc="lower center")

    return fig
