from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from tri_ct_tools.convert.geometry import calc_distances, cate_to_astra, d_through_column, pixel_coordinates
from tri_ct_tools.image.reader import singlecam_mean


def opt_ori_fun(shift, geoms_all_cams, row_start, row_end, cameras, det, img_all_cams):
    err = 0
    x_shift, y_shift, z_shift = shift
    rows = det['rows']
    cols = det['cols']
    for cam in cameras:
        (srcX, srcY, srcZ, dX, dY, dZ, uX, uY, uZ, vX, vY, vZ) = geoms_all_cams[cam]
        # Shift source coordinates
        srcX += x_shift
        srcY += y_shift
        srcZ += z_shift
        # Shift detector center coordinates
        dX += x_shift
        dY += y_shift
        dZ += z_shift

        x_coords, y_coords, z_coords = pixel_coordinates(dX, dY, dZ, uX, uY, uZ, vX, vY, vZ, rows, cols)

        d = d_through_column(x_coords, y_coords, z_coords, srcX, srcY, srcZ, det, diameter_type='inner')
        # d_mum = d * 10_000  # from cm to micrometer
        # d_mum_int = d_mum.astype(np.int32)
        # d = d_mum_int
        image_full, image_empty = img_all_cams[cam]

        d = d[row_start:row_end, :]
        image_empty = image_empty[row_start:row_end, :]
        image_full = image_full[row_start:row_end, :]

        d_mask = d > 0
        d = d[d_mask]
        image_empty = image_empty[d_mask]
        image_full = image_full[d_mask]

        # mean_distance = d.flatten()
        d = d.flatten()
        I_empty = image_empty.flatten()
        I_full = image_full.flatten()

        ln_intensity = -np.log(I_full/I_empty)

        sorted_index = np.argsort(d)

        ln_intensity_sorted = ln_intensity[sorted_index]
        diffs = np.diff(ln_intensity_sorted)

        err += np.sum(diffs**2)
    return err


def optimize_origin(geoms_all_cams, cameras, det, img_all_cams, row_start, row_end):
    x_ori_0 = [-0.1, 0.1, 0]
    bounds = [(-.5, .5) for _ in range(len(x_ori_0))]
    print(f'initial error: {opt_ori_fun(x_ori_0, geoms_all_cams, row_start, row_end, cameras, det, img_all_cams)}')
    res_ori = minimize(
        opt_ori_fun,
        x_ori_0,
        args=(geoms_all_cams, row_start, row_end, cameras, det, img_all_cams),
        method='Nelder-Mead',
        bounds=bounds,
        options={
            'disp': True,
            # 'eps': 1e-3,
            # 'ftol': 1e-2
            # 'return_all': True,
        })

    shift_ori_optim = res_ori.x

    print(f'geom optimized error: {opt_ori_fun(shift_ori_optim, geoms_all_cams,
                                               row_start, row_end, cameras, det,
                                               img_all_cams)}')
    return shift_ori_optim


def opt_det_fun(shift, geoms_all_cams, row_start, row_end, cameras, det, img_all_cams):
    err = 0
    camshifts = np.array(shift).reshape(3, 3)
    rows = det['rows']
    cols = det['cols']
    for cam in cameras:
        (srcX, srcY, srcZ, dX, dY, dZ, uX, uY, uZ, vX, vY, vZ) = geoms_all_cams[cam]

        d_xshift, d_yshift, d_zshift = camshifts[cam, :]
        # Shift detector center coordinates
        dX += d_xshift
        dY += d_yshift
        dZ += d_zshift

        x_coords, y_coords, z_coords = pixel_coordinates(dX, dY, dZ, uX, uY, uZ, vX, vY, vZ, rows, cols)

        d = d_through_column(x_coords, y_coords, z_coords, srcX, srcY, srcZ, det, diameter_type='inner')
        # d_mum = d * 10_000  # from cm to micrometer
        # d_mum_int = d_mum.astype(np.int32)
        # d = d_mum_int
        image_full, image_empty = img_all_cams[cam]

        d = d[row_start:row_end, :]
        image_empty = image_empty[row_start:row_end, :]
        image_full = image_full[row_start:row_end, :]

        d_mask = d > 0
        d = d[d_mask]
        image_empty = image_empty[d_mask]
        image_full = image_full[d_mask]

        # mean_distance = d.flatten()
        d = d.flatten()
        I_empty = image_empty.flatten()
        I_full = image_full.flatten()

        ln_intensity = -np.log(I_full/I_empty)

        sorted_index = np.argsort(d)

        ln_intensity_sorted = ln_intensity[sorted_index]
        diffs = np.diff(ln_intensity_sorted)

        err += np.sum(diffs**2)
    return err


def optimize_det(cameras, det, img_all_cams, row_start, row_end, geoms_ori_opt):
    # Shift detectors 1: XYZ, 2: XYZ, 3: XYZ
    x_det_0 = [
        0, 0, 0,
        0, 0, 0,
        0, 0, 0
    ]
    bounds = [(-.2, .2) for _ in range(len(x_det_0))]
    res_det = minimize(
        opt_det_fun,
        x_det_0,
        args=(geoms_ori_opt, row_start, row_end, cameras, det, img_all_cams),
        method='Nelder-Mead',
        bounds=bounds,
        options={
            'disp': True,
            # 'eps': 1e-3,
            # 'ftol': 1e-2
            # 'return_all': True,
        }
    )
    shift_det_optim = res_det.x
    print(f'det-optimized error: {opt_det_fun(shift_det_optim, geoms_ori_opt,
                                              row_start, row_end, cameras, det,
                                              img_all_cams)}')
    return shift_det_optim


def _opt_geom(geoms_all_cams, cameras, det, img_all_cams,
              window_height=10, verbosity="figure"):
    middle_line = det['rows'] // 2
    row_start = middle_line - (window_height // 2)
    row_end = middle_line + (window_height // 2)
    # shift origin (X, Y, Z)
    shift_ori_optim = optimize_origin(geoms_all_cams, cameras, det, img_all_cams,
                                      row_start, row_end)
    x_shift, y_shift, z_shift = shift_ori_optim

    geoms_ori_opt = geoms_all_cams
    for cam in cameras:
        (srcX, srcY, srcZ, dX, dY, dZ, uX, uY, uZ, vX, vY, vZ) = geoms_all_cams[cam]
        # Shift source coordinates
        srcX += x_shift
        srcY += y_shift
        srcZ += z_shift
        # Shift detector center coordinates
        dX += x_shift
        dY += y_shift
        dZ += z_shift
        geoms_ori_opt[cam] = [srcX, srcY, srcZ, dX, dY, dZ, uX, uY, uZ, vX, vY, vZ]

    shift_det_optim = optimize_det(cameras, det, img_all_cams, row_start, row_end, geoms_ori_opt)
    shift_det_optim = shift_det_optim.reshape(3, 3)

    print("Optimized shifts:")
    print(f"geom (XYZ): {shift_ori_optim}")
    print(f"det1 (XYZ): {shift_det_optim[0, :]}")
    print(f"det2 (XYZ): {shift_det_optim[1, :]}")
    print(f"det3 (XYZ): {shift_det_optim[2, :]}")

    geoms_det_opt = geoms_ori_opt
    for cam in cameras:
        (srcX, srcY, srcZ, dX, dY, dZ, uX, uY, uZ, vX, vY, vZ) = geoms_all_cams[cam]
        d_xshift, d_yshift, d_zshift = shift_det_optim[cam, :]
        # Shift detector center coordinates
        dX += d_xshift
        dY += d_yshift
        dZ += d_zshift
        geoms_det_opt[cam] = [srcX, srcY, srcZ, dX, dY, dZ, uX, uY, uZ, vX, vY, vZ]

    return geoms_det_opt


def geometry_optimizer(
        geom_path,
        img_path_base,
        full_img_folder,
        empty_img_folder,
        det,
        framerange=range(50, 200),
        cameras=range(0, 3)):

    img_shape = (det['cols'], det['rows'])
    geoms_all_cams = cate_to_astra(path=geom_path, det=det)

    img_all_cams = []
    for cam in cameras:
        path_full = img_path_base / full_img_folder / f"camera {cam+1}"
        path_empty = img_path_base / empty_img_folder / f"camera {cam+1}"
        img_full = singlecam_mean(path_full, framerange, img_shape)
        img_empty = singlecam_mean(path_empty, framerange, img_shape)
        img_all_cams.append((img_full, img_empty))

    geoms_optim = _opt_geom(geoms_all_cams, cameras, det, img_all_cams,
                            window_height=10)

    output_path = img_path_base / '00_calib'
    output_path.mkdir(exist_ok=True)

    for cam in cameras:
        d, _ = calc_distances(geoms_optim, cam, det)
        np.save(output_path / f'distances_cam{cam+1}.npy', d)
        path_full = img_path_base / full_img_folder / f"camera {cam+1}"
        path_empty = img_path_base / empty_img_folder / f"camera {cam+1}"
        img_full = singlecam_mean(path_full, framerange, img_shape)
        img_empty = singlecam_mean(path_empty, framerange, img_shape)

        ln_intensity = -np.log(img_full / img_empty)
        effective_attenuation = ln_intensity / d

        result = pd.DataFrame()
        result['distance_liquid'] = d.flatten()
        result['-ln_intensity'] = ln_intensity.flatten()
        result['effective_attenuation'] = effective_attenuation
        result.to_csv(output_path / f'intensity_cam{cam+1}.csv')

        metadata = {
            'src_full': path_full,
            'src_empty': path_empty
        }
        pd.DataFrame(metadata).to_csv(output_path / f'intensity_cam{cam+1}_metadata.csv')

    np.save(output_path / 'bhc_optimized_geom.npy', geoms_optim)

    print(f"Saved optimized geometry in {output_path}")
    return geoms_optim


if __name__ == "__main__":
    # path to geometry
    geom_path = Path(R'U:\Xray RPT ChemE\X-ray\Xray_data\2025-06-13 Rik Cropper'
                     R'\calib\NeedleCalibration_5degps\geom.npy')
    img_path_base = Path(R'u:\Xray RPT ChemE\X-ray\Xray_data\2025-06-13 Rik Cropper')
    full_img_path = '03_scattercorrected/WideCrop_Full_120kV_22Hz'
    empty_img_path = '03_scattercorrected/WideCrop_Empty_120kV_22Hz'

    det = {
        "rows": 1524,        # Number of rows in the detector
        "cols": 1548,        # Number of columns in the detector
        "pixel_width": 0.0198,      # Pixel width in cm
        "pixel_height": 0.0198,     # Pixel height in cm
        'det_width': 30.7,      # cm, detector width
        'det_height': 30.2,     # cm, detector height
        'column_inner_D': 19.2,     # cm
        'column_outer_D': 20.0      # cm
    }

    framerange = range(50, 200)
    cameras = range(0, 3)

    opt_geom = geometry_optimizer(geom_path, img_path_base, full_img_path,
                                  empty_img_path, det, framerange, cameras)
