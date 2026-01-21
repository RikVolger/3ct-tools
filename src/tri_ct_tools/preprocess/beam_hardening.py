import itertools
from pathlib import Path
import numpy as np

import matplotlib.pyplot as plt
import yaml

from tri_ct_tools.convert.geometry import calc_distances, cate_to_astra, plot_full_geom
from tri_ct_tools.image.reader import singlecam_mean
from tri_ct_tools.image.writer import array_to_tif
from tri_ct_tools.inspect.beam_hardening import single_cam_analysis


def BHC(I_BH, I_empty, coefficients, mu_eff, offset):
    # range of possible distances through liquid (for interpolation)
    x_values = np.linspace(0, 20, 1000)
    # y value to fit
    lnII = -np.log(I_BH/I_empty)

    # predicted y values based on polynomial and x-values
    P_x_values = np.polyval(coefficients, x_values)
    # use interpolation to determine ditance through liquid based on actual data
    x_fit_values = np.interp(lnII, P_x_values, x_values)

    I_noBH = np.exp(-mu_eff * x_fit_values - offset) * I_empty

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

    # mu_eff = np.sum(d[mask] * ln_rel_intensity[mask]) / np.sum(d[mask]**2)
    mu_eff = np.sum(ln_rel_intensity[mask]) / np.sum(d[mask])
    offset = coefficients[-1]
    mu_eff2 = (np.polyval(coefficients, mu_range) - offset) / mu_range
    print(f"original:  y = {mu_eff:.2f}.x")
    print(f"offsetted: y = {mu_eff2:.2f}.x + {offset:.2f}")

    return coefficients, mu_eff2, offset


def log_line_plot(
        path_full,
        path_empty,
        voltage,
        cam,
        geoms_all_cams,
        det,
        row_start=740,
        row_end=760):

    img_full = singlecam_mean(path_full, framerange, img_shape)
    img_empty = singlecam_mean(path_empty, framerange, img_shape)
    d, _ = calc_distances(geoms_all_cams, cam, det)

    # Take a horizontal slice to avoid probes screwing with the next steps
    assert row_end > row_start
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
    coeff, mu_eff, offset = calc_coef_mu(d[mask], rel_intensity_log)

    # Perform beam hardening correction
    img_full_BHC = BHC(img_full, img_empty, coeff, mu_eff, offset)
    rel_intensity_noBH = img_full_BHC / img_empty

    mask = rel_intensity_noBH > 0
    rel_intensity_log_noBH = -np.log(rel_intensity_noBH[mask])

    # Plot the results
    d_vals = np.linspace(0, d.max(), 100)
    p_fit = np.polyval(coeff, d_vals)

    fig, ax = plt.subplots(1, 1)
    ax.plot(d[mask], rel_intensity_log, 'o', alpha=0.1, label='data')
    ax.plot(d[mask], rel_intensity_log_noBH, 'o', alpha=0.1, label='data-BHC')
    ax.plot(d_vals, p_fit, label='polyfit')
    ax.plot(d_vals, d_vals*mu_eff + offset, label='constant attenuation')
    ax.set_title(
        f"Camera {cam+1}, {voltage} kV"
        R" - $\mu_{eff}$ = "
        f"{mu_eff:.3f}"
    )
    ax.set_ylabel("Attenuation as $-ln I_{full}/I_{empty}$")
    ax.set_xlabel("Pathlength (cm)")
    ax.legend()
    return fig


def beam_hardening_coefficients(d, img_full, img_empty):
    # Take a horizontal slice to avoid probes screwing with the next steps
    # row_start = 300
    # row_end = 1100
    # d = d[row_start:row_end, :]
    # img_full = img_full[row_start:row_end, :]
    # img_empty = img_empty[row_start:row_end, :]

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
    coeff, mu_eff, offset = calc_coef_mu(d[mask], rel_intensity_log,    5)
    return coeff, mu_eff, offset


def get_coefficients(det, ROI, geoms_all_cams, cam, img_full, img_empty):
    d, _ = calc_distances(geoms_all_cams, cam, det)

    # Crop out probes and tubes
    img_full = img_full[ROI[0]:ROI[1], :]
    img_empty = img_empty[ROI[0]:ROI[1], :]
    d = d[ROI[0]:ROI[1], :]

    coeff, mu_eff, offset = beam_hardening_coefficients(d, img_full, img_empty)
    return coeff, mu_eff, offset


if __name__ == "__main__":

    input_file = Path("inputs/beam_hardening_corrections.yaml")
    with open(input_file) as bhc_yaml:
        bhc_input = yaml.safe_load(bhc_yaml)

    det = bhc_input['det']
    # path to geometry
    geom_path = Path(bhc_input['geom_path'])
    framerange = {
        'full': range(bhc_input['framerange']['full']['start'],
                      bhc_input['framerange']['full']['stop'],
                      bhc_input['framerange']['full']['step']),
        'empty': range(bhc_input['framerange']['empty']['start'],
                       bhc_input['framerange']['empty']['stop'],
                       bhc_input['framerange']['empty']['step']),
        'dark': range(bhc_input['framerange']['dark']['start'],
                      bhc_input['framerange']['dark']['stop'],
                      bhc_input['framerange']['dark']['step']),
        'meas': range(bhc_input['framerange']['measurement']['start'],
                      bhc_input['framerange']['measurement']['stop'],
                      bhc_input['framerange']['measurement']['step']),
    }
    img_shape = (det['cols'], det['rows'])
    ROI = bhc_input['ROI']
    cameras = bhc_input['cameras']

    # If the geom path points to a file from the bhc optimization, it is not
    # cate-like, and can be loaded in a simplified manner
    if geom_path.name == "bhc_optimized_geom.npy":
        geoms_all_cams = np.load(geom_path)
    else:
        geoms_all_cams = cate_to_astra(path=geom_path, det=det)

    plot_full_geom(geoms_all_cams, det)
    plt.show()

    series = bhc_input['series']

    for s in series:
        name = s['name']
        full_path = Path(s['full'])
        empty_path = Path(s['empty'])
        if 'dark' in s.keys() and s['dark'] is not None:
            dark_path = Path(s['dark'])
        else:
            dark_path = None
            img_dark = None
        empty_copy_path = Path(s['empty_copy'])

        print(f"\nRunning beam hardening correction for {name}")

        for meas, cam in itertools.product(s['meas'], cameras):
            if dark_path is not None:
                dark_path_cam = dark_path / f"camera {cam+1}"
                img_dark = singlecam_mean(dark_path_cam, framerange['dark'], img_shape)
            full_path_cam = full_path / f"camera {cam+1}"
            empty_path_cam = empty_path / f"camera {cam+1}"
            meas_path_cam = Path(meas['input']) / f"camera {cam+1}"
            img_full = singlecam_mean(full_path_cam, framerange['full'], img_shape, img_dark)
            img_empty = singlecam_mean(empty_path_cam, framerange['empty'], img_shape, img_dark)
            if "Full" in str(meas_path_cam):
                img_meas = singlecam_mean(meas_path_cam, framerange['full'], img_shape, img_dark)
            else:
                img_meas = singlecam_mean(meas_path_cam, framerange['meas'], img_shape, img_dark)
            coeff, mu_eff, offset = get_coefficients(det, ROI, geoms_all_cams,
                                                     cam, img_full, img_empty)

            meas_bhc = BHC(img_meas, img_empty, coeff, mu_eff, offset)

            meas_output_path = Path(meas['output'])
            meas_output_cam = meas_output_path / f"camera {cam+1}"
            array_to_tif(meas_bhc.astype(np.int32), meas_output_cam, 'average.tif')
            bhc_coefficients = {
                'mu_eff': mu_eff,
                'offset': offset,
                'poly_coefficients': coeff
            }
            # Even though BHC does nothing on empty, want to have it in the same folder.
            array_to_tif(img_empty, empty_copy_path / f"camera {cam+1}", 'average.tif')

            output_file = meas_output_cam / f'bhc_coefficients_cam{cam+1}.yaml'
            with open(output_file, 'w') as outfile:
                yaml.dump(bhc_coefficients, outfile, default_flow_style=False)
