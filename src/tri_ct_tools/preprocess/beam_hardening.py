from pathlib import Path
import numpy as np

import matplotlib.pyplot as plt

from tri_ct_tools.convert.geometry import cate_to_astra, plot_full_geom
from tri_ct_tools.image.reader import singlecam_mean
from tri_ct_tools.inspect.beam_hardening import single_cam_analysis


def BHC(I_BH, I_empty, coefficients, mu_eff):
    # range of possible distances through liquid (for interpolation)
    x_values = np.linspace(0, 20, 2000)
    # y value to fit
    lnII = -np.log(I_BH/I_empty)

    # predicted y values based on polynomial and x-values
    P_x_values = np.polyval(coefficients, x_values)
    # use interpolation to determine ditacne through liquid based on actual data
    x_fit_values = np.interp(lnII, P_x_values, x_values)

    I_noBH = np.exp(-mu_eff * x_fit_values) * I_empty

    return I_noBH


def calc_coef_mu(d, ln_rel_intensity, mu_range=3, poly_degree=3):

    coefficients = np.polyfit(d, ln_rel_intensity, poly_degree)
    # mu calculation based on pathlengths under 3 cm
    mask = d < mu_range

    mu_eff = np.sum(d[mask] * ln_rel_intensity[mask]) / np.sum(d[mask]**2)

    return coefficients, mu_eff


if __name__ == "__main__":
    # path to geometry
    geom_path = Path(R'u:\Xray RPT ChemE\X-ray\Xray_data\2025-06-13 Rik Cropper\00_calib\bhc_optimized_geom.npy')
    output_path = geom_path.parent / "bhc"
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
    geoms_all_cams = cate_to_astra(path=geom_path, det=det)

    img_path_base = Path(R'u:\Xray RPT ChemE\X-ray\Xray_data\2025-06-13 Rik Cropper')

    for cam in cameras:
        # paths to camera folders
        path_full = img_path_base / f'03_scattercorrected/WideCrop_Full_120kV_22Hz/camera {cam+1}'
        path_empty = img_path_base / f'03_scattercorrected/WideCrop_Empty_120kV_22Hz/camera {cam+1}'
        img_full = singlecam_mean(path_full, framerange, img_shape)
        img_empty = singlecam_mean(path_empty, framerange, img_shape)
        images = (img_full, img_empty)
        # calculate distances
        single_cam_analysis(geoms_all_cams, images, cam, det, output_path)

    plot_full_geom(geoms_all_cams, det)

    calc_coef_mu(left_half_x, ln_left)

    plt.show()
