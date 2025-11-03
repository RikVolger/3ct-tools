import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import pandas as pd
from scipy.optimize import minimize, brute
from tri_ct_tools.image.reader import singlecam_mean


def cate_to_astra(path, det, geom_scaling_factor=None, angles=None):
    """Convert `Geometry` objects from our calibration package to the
    ASTRA vector convention."""

    import pickle
    from cate import astra, xray
    from numpy.lib.format import read_magic, read_array_header_1_0, read_array_header_2_0

    class RenamingUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if name == "StaticGeometry":
                name = "Geometry"
            return super().find_class(module, name)

    with open(path, "rb") as fp:
        version = read_magic(fp)
        if version[0] == 1:
            dtype = read_array_header_1_0(fp)[2]
        elif version[0] == 2:
            dtype = read_array_header_2_0(fp)[2]
        # _check_version(version)
        # dtype = _read_array_header(fp, version)[2]
        assert dtype.hasobject
        multicam_geom = RenamingUnpickler(fp).load()[0]

    detector = astra.Detector(
        det["rows"], det["cols"], det["pixel_width"], det["pixel_height"]
    )

    def _to_astra_vec(g):
        v = astra.geom2astravec(g, detector.todict())
        if geom_scaling_factor is not None:
            v = np.array(v) * geom_scaling_factor
        return v

    if angles is None:
        geoms = []
        for _, g in sorted(multicam_geom.items()):
            geoms.append(_to_astra_vec(g))
        return geoms
    else:
        geoms_all_cams = {}
        for cam in list(multicam_geom.keys()):
            geoms = []
            for a in angles:
                g = xray.transform(multicam_geom[cam], yaw=a)
                geoms.append(_to_astra_vec(g))
            geoms_all_cams[cam] = geoms

        return geoms_all_cams


def pixel_coordinates(dX, dY, dZ, uX, uY, uZ, vX, vY, vZ, rows, cols):
    # Create a 2D grid of row indices (i) and column indices (j)
    # Shape (rows, 1) the middle of the detector is at (0,0), so substract half the number of rows
    i_vals = np.arange(-rows // 2, rows // 2).reshape(-1, 1)
    # Shape (1, cols) the middle of the detector is at (0,0), so substract half the number of cols
    j_vals = np.arange(-cols // 2, cols // 2).reshape(1, -1)
    # print(i_vals)

    # Compute the 3D coordinates for all pixels in centimeters
    # coordinate of the detector middle corrected for every pixel based on vector describing distance between pixels
    x = (dX + j_vals * uX + i_vals * vX)
    y = (dY + j_vals * uY + i_vals * vY)
    z = (dZ + j_vals * uZ + i_vals * vZ)

    return x, y, z


def d_through_column(x_coords, y_coords, z_coords, srcX, srcY, srcZ, det, diameter_type='inner'):
    # Directional vector is build up of a, b and c
    a = x_coords - srcX     # in x direction
    b = y_coords - srcY     # in y direction
    c = z_coords - srcZ     # in z direction

    # Equations for all the straight lines connecting the X-ray source and the pixel on the detector
    # x = srcX + a*t
    # y = srcY + b*t
    # z = srcZ + c*t
    #
    # Equation for cylinder:
    # x**2 + y**2 = r**2
    if diameter_type == 'inner':
        D = det['column_inner_D']
    elif diameter_type == 'outer':
        D = det['column_outer_D']

    r = D / 2   # cm

    # Coefficients for the quadratic equation
    # (see notes: write equation as quadratic equation for t. so:
    #  A*t**2 + B*t + C = 0)
    A = a**2 + b**2
    B = 2 * (srcX * a + srcY * b)
    C = srcX**2 + srcY**2 - r**2

    # discriminant
    Disc = B**2 - 4 * A * C

    # Handle cases where the discriminant is negative (no real solution)
    Disc[Disc < 0] = 0

    # Two possible solutions for t
    t1 = (-B + np.sqrt(Disc)) / (2 * A)
    t2 = (-B - np.sqrt(Disc)) / (2 * A)

    # coordinates of first intercept through cylinder
    x_A = srcX + a * t1
    y_A = srcY + b * t1
    z_A = srcZ + c * t1

    # Coordinates of second intercept through cylinder
    x_B = srcX + a * t2
    y_B = srcY + b * t2
    z_B = srcZ + c * t2

    # Calculate the distance through the column
    d = np.sqrt((x_B - x_A)**2 + (y_B - y_A)**2 + (z_B - z_A)**2)

    return d


def plot_full_geom(geom_all_cams, det):
    num_cam = len(geom_all_cams)    # number of cameras
    fig, ax = plt.subplots()

    circle = plt.Circle((0, 0), det['column_outer_D'] / 2, fill=False)
    ax.add_patch(circle)
    for i in range(num_cam):

        (srcX, srcY, srcZ, dX, dY, dZ, uX, uY, uZ, vX, vY, vZ) = geom_all_cams[i]

        x_coords, y_coords, z_coords = pixel_coordinates(
            dX, dY, dZ,
            uX, uY, uZ,
            vX, vY, vZ,
            det['rows'], det['cols']
        )

        ax.scatter(x_coords[0, :], y_coords[0, :])
        ax.scatter([srcX, dX], [srcY, dY], label=f"S-D {i + 1}", alpha=0.5)
        ax.plot([x_coords[-1, -1], srcX, x_coords[0, 0]], [y_coords[-1, -1], srcY, y_coords[0, 0]], label="corner_line")

    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Geometry of system')
    ax.legend(loc='upper right')
    plt.show()
    print('Plotted geometry of system with succes')
    return


def calc_distances(geoms_all_cams, cam, det) -> np.ndarray:
    rows, cols = det['rows'], det['cols']
    (srcX, srcY, srcZ, dX, dY, dZ, uX, uY, uZ, vX, vY, vZ) = geoms_all_cams[cam]

    x_coords, y_coords, z_coords = pixel_coordinates(dX, dY, dZ, uX, uY, uZ, vX, vY, vZ, rows, cols)
    
    d = d_through_column(x_coords, y_coords, z_coords, srcX, srcY, srcZ, det, diameter_type='inner')
    d_outer = d_through_column(x_coords, y_coords, z_coords, srcX, srcY, srcZ, det, diameter_type='outer')

    return d, d_outer


def single_cam_analysis(geoms_all_cams, images, cam, det, output_path, img_offset=0):
    d, d_outer = calc_distances(geoms_all_cams, cam, det)

    np.save(output_path / f'distances_cam{cam+1}.npy', d)

    image_full, image_empty = images

    row_start = 720
    row_end = 780
    # row_start = 1
    # row_end = -1

    # col_start = 1
    # col_end = -1
    if cam == 0:
        col_start = 40
        col_end = 1450
        # img_offset = -10
    elif cam == 1:
        col_start = 100
        col_end = 1520
        # img_offset = 9
    elif cam == 2:
        col_start = 60
        col_end = 1480
        # img_offset = -1

    col_start_d = col_start
    col_start_img = col_start + img_offset
    col_end_d = col_end
    col_end_img = col_end + img_offset
    
    d = d[row_start:row_end, col_start_d:col_end_d]
    image_empty = image_empty[row_start:row_end, col_start_img:col_end_img]
    image_full = image_full[row_start:row_end, col_start_img:col_end_img]

    multichannel_im = np.zeros((*image_full.shape, 3))
    multichannel_im[..., 0] = image_full / image_full.max()
    multichannel_im[..., 1] = d / d.max()

    image_norm_inv = d.max() - (image_full / image_full.max() * d.max())
    image_full_inv = image_full.max() - image_full

    image_empty_inv = image_empty.max() - image_empty

    ln_intensity = -np.log(image_full / image_empty)

    rel_img = image_full / image_empty
    rel_img_inv = rel_img.max() - rel_img
    rel_img_inv_norm = rel_img / rel_img.max() * image_full_inv.max()
    # rel_img_norm = rel_img / rel_img.max() * d.max()
    hline = 750 - row_start
    vline = 750 - col_start
    # multichannel_im[..., 2] = d_outer / d_outer.max()
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

    # fig, ax = plt.subplots()
    # ax.twinx()
    # return
    # distance through water (inner diameter)
    mean_distance = d.flatten()

    # mean_distance_outer = d_outer[row_start:row_end, col_start:col_end].mean(axis=0).flatten() # through water+column (outer diameter)
    # mean_distance_outer = d_outer[row_start:row_end, col_start:col_end].flatten() # through water+column (outer diameter)

    # distance_through_column = mean_distance_outer - mean_distance

    # ratio_column_water = distance_through_column / mean_distance_outer

    I_empty = image_empty.flatten()
    I_full = image_full.flatten()

    ln_intensity = -np.log(I_full/I_empty)
    effective_attenuation = ln_intensity / mean_distance

    result = pd.DataFrame()
    result['distance_liquid'] = mean_distance
    # result['distance_column'] = distance_through_column
    # result['ratio'] = ratio_column_water
    result['-ln_intensity'] = ln_intensity
    result['effective_attenuation'] = effective_attenuation
    result['I_empty'] = I_empty
    result['I_full'] = I_full
    result.to_csv(output_path / 'intensity.csv')

    mask = mean_distance > 0
    fig, ax = plt.subplots(1, 2, figsize=(15, 5), layout='tight')
    fig.suptitle(f"Detector {cam+1}, geom {cam+1}")
    ax[0].plot(mean_distance[mask], ln_intensity[mask], '--o', alpha=0.5)
    ax[0].set_xlabel('x (cm)', fontsize=15)
    ax[0].set_ylabel(R'$-ln(\frac{I(x)}{I_{empty}})$', fontsize=15)

    ax[1].plot(mean_distance[mask], effective_attenuation[mask], '--o', alpha=0.5)
    ax[1].set_xlabel('x (cm)', fontsize=15)
    ax[1].set_ylabel(R'$\mu_{eff}$'+' (1/cm)', fontsize=15)

    # fig, ax = plt.subplots(1, 1, layout='tight')
    # ax.plot(mean_distance, ratio_column_water)
    # ax.set_ylabel(r'$x_{wall}/x_{liquid}$')
    # ax.set_xlabel(r'$x_{liquid}$' + '(cm)')

    # Define the value range
    min_value = 16000
    max_value = 20000

    # col_start = 1
    # col_end = -1
    # row_start = 1
    # row_end = -1

    # Find the locations where the values fall within the range
    points = np.where(
        (image_full[row_start:row_end, col_start:col_end] >= min_value) &
        (image_full[row_start:row_end, col_start:col_end] <= max_value))
    no_column = np.where((d[row_start:row_end, col_start:col_end] == 0))

    fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].imshow(image_full[row_start:row_end, col_start:col_end])

    # Overlay red dots on the points where the values are between 2500 and 6000
    ax[0].scatter(points[1], points[0], color='red', marker='o', s=5)  # Red dots for values in range
    ax[0].scatter(no_column[1], no_column[0], color='blue', marker='o', s=5, alpha=0.1)
    ax[0].set_aspect('equal')

    ax[1].imshow(image_full)
    ax[1].set_aspect('equal')


def optimize_geometry(geoms_all_cams, cameras, det, img_all_cams, verbosity="figure"):
    row_start = 749
    row_end = 751
    x0 = [-0.1, 0.1, 0]
    print(f'initial error: {opt_geom_fun(x0, geoms_all_cams, row_start, row_end, cameras, det, img_all_cams)}')
    res = minimize(
        opt_geom_fun,
        x0,
        args=(geoms_all_cams, row_start, row_end, cameras, det, img_all_cams),
        method='Nelder-Mead',
        bounds=((-.5, .5), (-.5, .5), (-.5, .5)),
        options={
            'disp': True,
            'return_all': True,
        })
    # res = brute(
    #     opt_geom_fun,
    #     ((-.5, .5), (-.5, .5), (-.5, .5)),
    #     args=(geoms_all_cams, row_start, row_end, cameras, det, img_all_cams),
    #     Ns=10
    # )
    shift_optim = res.x
    x_shift, y_shift, z_shift = shift_optim
    print(f"Optimized shifts: {shift_optim}")

    geoms_opt = geoms_all_cams
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
        geoms_opt[cam] = [srcX, srcY, srcZ, dX, dY, dZ, uX, uY, uZ, vX, vY, vZ]

        if verbosity == "figure":
            single_cam_analysis(geoms_opt, img_all_cams[cam], cam, det, Path('output'))
    plt.show()
    return geoms_opt


def opt_geom_fun(shift, geoms_all_cams, row_start, row_end, cameras, det, img_all_cams):
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
        d_mum = d * 10_000
        d_mm_int = d_mum.astype(np.int32)
        image_full, image_empty = img_all_cams[cam]

        d = d_mm_int[row_start:row_end, :]
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


def load_images(det, path_full, path_empty):
    average_file_full = path_full / 'average.tif'
    image_full = read_average(det, path_full, average_file_full)

    average_file_empty = path_empty / 'average.tif'
    image_empty = read_average(det, path_empty, average_file_empty)

    return image_full, image_empty


def read_average(det, path, avg_file):
    if avg_file.exists():
        img = np.array(Image.open(avg_file), dtype=np.float64)
    else:
        img = singlecam_mean(path, range(50, 200), (det['cols'], det['rows']))
    return img


if __name__ == "__main__":
    # path to geometry
    geom_path = Path(R'U:\Xray RPT ChemE\X-ray\Xray_data\2025-06-13 Rik Cropper\calib\NeedleCalibration_5degps\geom.npy')
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

    cameras = range(0, 3)

    geoms_all_cams = cate_to_astra(path=geom_path, det=det)

    img_path_base = Path(R'u:\Xray RPT ChemE\X-ray\Xray_data\2025-06-13 Rik Cropper')
    img_all_cams = []
    for cam in cameras:
        path_full = img_path_base / '03_scattercorrected/WideCrop_Full_120kV_22Hz' / f"camera {cam+1}"
        path_empty = img_path_base / '03_scattercorrected/WideCrop_Empty_120kV_22Hz' / f"camera {cam+1}"

        img_all_cams.append(load_images(det, path_full, path_empty))
    
    geoms_optim = optimize_geometry(geoms_all_cams, cameras, det, img_all_cams)

    for cam in cameras:
        # paths to camera folders
        path_full = Path(R'u:\Xray RPT ChemE\X-ray\Xray_data\2025-06-13 Rik Cropper\03_scattercorrected\WideCrop_Full_120kV_22Hz', f"camera {cam+1}")
        path_empty = Path(R'u:\Xray RPT ChemE\X-ray\Xray_data\2025-06-13 Rik Cropper\03_scattercorrected\WideCrop_Empty_120kV_22Hz', f"camera {cam+1}")

        images = load_images(det, path_full, path_empty)
        # calculate distances
        single_cam_analysis(geoms_all_cams, images, cam, det, output_path)

    plot_full_geom(geoms_all_cams, det)

    plt.show()
