import numpy as np
import matplotlib.pyplot as plt


def cate_to_astra(path, det, geom_scaling_factor=None, angles=None):
    """Convert Geometry objects from calibration package to ASTRA vector convention.

    Loads a pickled geometry object from the CATE calibration package and converts
    it to ASTRA's vector format. Optionally applies geometric scaling and can transform
    geometries for multiple viewing angles.

    Args:
        path (str | Path): Path to the pickled geometry file.
        det (dict): Detector configuration with keys 'rows', 'cols', 'pixel_width',
            'pixel_height'.
        geom_scaling_factor (float | None, optional): Factor to scale geometry vectors.
            Defaults to None.
        angles (list | None, optional): List of rotation angles (in degrees) to apply
            to geometries. If None, only returns geometries without rotation.
            Defaults to None.

    Returns:
        list | dict: If angles is None, returns a list of geometry vectors.
                    If angles is provided, returns a dict with camera keys mapping
                    to lists of geometry vectors for each angle.
    """
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
    """Calculate 3D coordinates of all detector pixels.

    Computes the 3D carthesian coordinates (X, Y, Z) for every pixel in the detector
    based on the detector center position and pixel spacing vectors.

    Args:
        dX, dY, dZ (float): Detector center position coordinates (cm).
        uX, uY, uZ (float): Detector u-axis direction (horizontal spacing between pixels).
        vX, vY, vZ (float): Detector v-axis direction (vertical spacing between pixels).
        rows (int): Number of detector rows.
        cols (int): Number of detector columns.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: Arrays x, y, z of shape (rows, cols)
            containing the 3D coordinates of each pixel.
    """
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


def d_through_column(x_coords, y_coords, z_coords, srcX, srcY, srcZ, det,
                     diameter_type='inner') -> np.ndarray:
    """Calculate the distance a ray travels through a cylindrical column.

    For each pixel on the detector, computes the distance traveled by the X-ray
    from the source through the cylindrical column to the detector pixel.

    Args:
        x_coords, y_coords, z_coords (np.ndarray): Detector pixel coordinates (cm).
        srcX, srcY, srcZ (float): X-ray source coordinates (cm).
        det (dict): Detector configuration with 'column_inner_D' and/or
            'column_outer_D' keys for inner and outer diameter (cm).
        diameter_type (str, optional): Which column diameter to use: 'inner' or 'outer'.
            Defaults to 'inner'.

    Returns:
        np.ndarray: 2D array of distances (cm) with same shape as input coordinates.
    """
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
    """Plot the full geometry of the multi-camera setup.

    Visualizes the positions of all cameras, X-ray sources, detectors, and the
    cylindrical column in a 2D top-down view.

    Args:
        geom_all_cams (list): List of geometry vectors (tuples of 12 float values)
            for each camera.
        det (dict): Detector configuration with keys 'column_outer_D' (outer diameter
            of the column in cm), 'rows', 'cols'.

    Returns:
        None: Displays the plot using matplotlib.
    """
    num_cam = len(geom_all_cams)
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
    """Calculate distances through column for a specific camera.

    Computes both the inner and outer distances through the cylindrical column
    for all pixels in the specified camera's detector.

    Args:
        geoms_all_cams (list): List of geometry vectors for all cameras.
        cam (int): Camera index to calculate distances for.
        det (dict): Detector configuration with required keys.

    Returns:
        tuple[np.ndarray, np.ndarray]: Tuple of (d_inner, d_outer) arrays, each
            of shape (rows, cols) containing distances in cm.
    """
    rows, cols = det['rows'], det['cols']
    (srcX, srcY, srcZ, dX, dY, dZ, uX, uY, uZ, vX, vY, vZ) = geoms_all_cams[cam]

    x_coords, y_coords, z_coords = pixel_coordinates(dX, dY, dZ, uX, uY, uZ, vX, vY, vZ, rows, cols)

    d = d_through_column(x_coords, y_coords, z_coords, srcX, srcY, srcZ, det, diameter_type='inner')
    d_outer = d_through_column(x_coords, y_coords, z_coords, srcX, srcY, srcZ, det, diameter_type='outer')

    return d, d_outer
