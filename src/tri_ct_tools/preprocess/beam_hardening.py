import itertools
from pathlib import Path
import numpy as np

import matplotlib.pyplot as plt
import yaml

from tri_ct_tools.convert.geometry import calc_distances, cate_to_astra, plot_full_geom
from tri_ct_tools.image.reader import singlecam_mean
from tri_ct_tools.image.writer import array_to_tif
from tri_ct_tools.inspect.beam_hardening import single_cam_analysis


def BHC(I_BH, I_empty, coefficients, mu_eff):
    # range of possible distances through liquid (for interpolation)
    x_values = np.linspace(0, 20, 1000)
    # y value to fit
    lnII = -np.log(I_BH/I_empty)

    # predicted y values based on polynomial and x-values
    P_x_values = np.polyval(coefficients, x_values)
    # use interpolation to determine ditance through liquid based on actual data
    x_fit_values = np.interp(lnII, P_x_values, x_values)

    I_noBH = np.exp(-mu_eff * x_fit_values) * I_empty

    # After beam hardening corrections, nan values can appear in image. Interpolate.
    if np.any(np.isnan(I_noBH)):
        nanmask = np.isnan(I_noBH)
        I_noBH[nanmask] = np.interp(
            np.flatnonzero(nanmask),
            np.flatnonzero(~nanmask),
            I_noBH[~nanmask]
        )

    return I_noBH


def calc_coef_mu(d, ln_rel_intensity, mu_range=3, poly_degree=3):
    """Fit a polynomial and attenuation value through the intensity-distance data

    Args:
        d (np.ndarray): Source-detector distance (cm) through water for each pixel
        ln_rel_intensity (np.ndarray): Negative logarithm of I_full / I_empty
        mu_range (int, optional): Distance (cm) over which to calculate 'default'
            attenuation coefficient. Defaults to 3.
        poly_degree (int, optional): Degree of the polynomial to fit. Defaults to 3.

    Returns:
        Tuple[np.ndarray, float]: The coefficients of the fitted polynomial and 
            the effective attenuation coefficient
    """
    coefficients = np.polyfit(d, ln_rel_intensity, poly_degree)
    # mu calculation based on pathlengths under 3 cm
    mask = d < mu_range

    mu_eff = np.sum(d[mask] * ln_rel_intensity[mask]) / np.sum(d[mask]**2)

    return coefficients, mu_eff


if __name__ == "__main__":
    # path to geometry
    geom_path = Path(R'u:\Xray RPT ChemE\X-ray\Xray_data\2025-06-13 Rik Cropper\00_calib\bhc_optimized_geom.npy')
    output_path = geom_path.parent.parent / "04_bhcorrected"
    output_path.mkdir(exist_ok=True)

    det = {
        "rows": 1524,        # Number of rows in the detector
        "cols": 1548,        # Number of columns in the detector
        "pixel_width": 0.0198,      # Pixel width in cm
        "pixel_height": 0.0198,     # Pixel height in cm
        'det_width': 30.7,      # cm, detector width
        'det_height': 30.2,     # cm, detector height
        'column_inner_D': 19.2,   # cm
        'column_outer_D': 20.0  # cm
    }
    img_shape = (det['cols'], det['rows'])
    framerange = range(50, 200)

    cameras = range(0, 3)

    # [ ] If the geom path points to a file from the bhc optimization, it is not
    # cate-like, and should be loaded in a simplified manner. Need to devise a
    # check for that.
    if geom_path.name == "bhc_optimized_geom.npy":
        geoms_all_cams = np.load(geom_path)
    else:
        geoms_all_cams = cate_to_astra(path=geom_path, det=det)

    img_path_base = Path(R'u:\Xray RPT ChemE\X-ray\Xray_data\2025-06-13 Rik Cropper')

    plot_full_geom(geoms_all_cams, det)
    plt.show()
    # for cam in cameras:
    #     # paths to camera folders
    #     path_full = img_path_base / f'03_scattercorrected/WideCrop_Full_120kV_22Hz/camera {cam+1}'
    #     path_empty = img_path_base / f'03_scattercorrected/WideCrop_Empty_120kV_22Hz/camera {cam+1}'
    #     img_full = singlecam_mean(path_full, framerange, img_shape)
    #     img_empty = singlecam_mean(path_empty, framerange, img_shape)
    #     images = (img_full, img_empty)
    #     # calculate distances
    #     single_cam_analysis(geoms_all_cams, images, cam, det)
    # plt.show()
    for cam in cameras:
        path_full = img_path_base / f'03_scattercorrected/WideCrop_Full_120kV_22Hz/camera {cam+1}'
        path_empty = img_path_base / f'03_scattercorrected/WideCrop_Empty_120kV_22Hz/camera {cam+1}'
        img_full = singlecam_mean(path_full, framerange, img_shape)
        img_empty = singlecam_mean(path_empty, framerange, img_shape)
        images = (img_full, img_empty)
        d, _ = calc_distances(geoms_all_cams, cam, det)
        # Take a horizontal slice to avoid probes screwing with the next steps
        row_start = 740
        row_end = 760
        d = d[row_start:row_end, :]
        img_full = img_full[row_start:row_end, :]
        img_empty = img_empty[row_start:row_end, :]
        # Only consider values where the beam passed through the column
        d_mask = d > 0
        d = d[d_mask]
        img_full = img_full[d_mask]
        img_empty = img_empty[d_mask]
        # Calculate relative intensity
        rel_intensity = img_full / img_empty
        # Clean up to avoid problems in logarithm
        mask = rel_intensity > 0
        rel_intensity_log = -np.log(rel_intensity[mask])
        # Fit polynomial to the data
        coeff, mu_eff = calc_coef_mu(d[mask], rel_intensity_log, 7)
        # Perform beam hardening correction
        img_full_noBH = BHC(img_full, img_empty, coeff, mu_eff)
        rel_intensity_noBH = img_full_noBH / img_empty

        mask = rel_intensity_noBH > 0
        rel_intensity_log_noBH = -np.log(rel_intensity_noBH[mask])
        # Plot the results
        d_vals = np.linspace(0, d.max(), 100)
        p_fit = np.polyval(coeff, d_vals)

        fig, ax = plt.subplots(1, 1)
        ax.plot(d[mask], rel_intensity_log, 'o', alpha=0.1, label='data')
        ax.plot(d[mask], rel_intensity_log_noBH, 'o', alpha=0.1, label='data-BHC')
        ax.plot(d_vals, p_fit, label='polyfit')
        ax.plot(d_vals, d_vals*mu_eff, label='constant attenuation')
        ax.set_title(
            f"Attenuation plot for cam {cam+1}"
            R" - $\mu_{eff}$ = "
            f"{mu_eff:.3f}"
        )
        ax.legend()

        output_path = img_path_base / '00_calib'
    plt.show()
