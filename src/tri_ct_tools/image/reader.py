from pathlib import Path

import numpy as np
import tifffile


def print_reading(file):
    print(f"Reading {file}")


def single(img_file=None, img_folder=None, frame=None):
    """Load and return a single image

    Args:
        img_file (None or str or pathlib.Path): Image to be read, including extension
        img_folder (None or pathlib.Path): 'camera n' folder containing images. Requires
            defined `frame`.
        frame (None or sr): Frame of interest. If used, `img_folder` should be defined.

    Raises:
        ValueError: An invalid combination of arguments is provided.

    Returns:
        np.ndarray: Image file as Numpy NDArray
    """
    if img_file is not None:
        print_reading(img_file)
        img = tifffile.imread(img_file)
        return img

    if None not in [img_folder, frame]:
        img_file = img_folder / f"img_{frame}.tif"
        print_reading(img_file)
        img = tifffile.imread(img_file)
        return img

    print("""Invalid use of read `read_single`. Provide either:\n
          - img_file: an image to be read
          - img_folder & frame
          """)
    raise ValueError


def singlecam_mean(cam_folder: Path, frames, img_shape, dark=None):
    """Read the frames in cam_folder, and return a time-averaged image

    Args:
        cam_folder (pathlib.Path): Folder containing tiff image files
        frames (iterable): Frame numbers to read
        img_shape (list or tuple): Image shape in [width, height] format
        dark (np.ndarray, optional): Dark image to be subtracted from the averaged
            image. Defaults to None.

    Returns:
        np.ndarray: 2D Array of size (width, height) containing the time-averaged image
    """
    img = np.zeros(img_shape)
    print_reading(cam_folder)
    average_file = cam_folder / 'average.tif'
    if average_file.exists():
        img = tifffile.imread(average_file)
    else:
        img_files = [cam_folder / f"img_{fr}.tif" for fr in frames]
        img = tifffile.imread(img_files).mean(axis=0)
    if dark is not None:
        img -= dark
    return img


def singlecam_series(cam_folder, frames, img_shape, dark=None):
    """Read images from a single camera folder and return all indicated frames

    Args:
        cam_folder (pathlib.Path): Folder containing .tif image files
        frames (iterable): Frames to be read
        img_shape (list or tuple): Image shape in [width, height] format
        dark (np.ndarray, optional): Dark image to be subtracted from the averaged
            image. Defaults to None.

    Returns:
        np.ndarray: 3D array of size (n_frames, width, height) containing images
    """
    scan_img = np.zeros((len(frames), *img_shape))
    print_reading(cam_folder)
    img_files = [cam_folder / f"img_{fr}.tif" for fr in frames]
    scan_img[:, :, :] = tifffile.imread(img_files)
    if dark is not None:
        scan_img -= dark
    return scan_img


def multicam_mean(folder: Path, cameras, frames, img_shape, dark=None):
    """Read images from multiple cameras, and return the time-averaged image

    Args:
        folder (pathlib.Path): Path to the folder containing the `camera n` folders
        cameras (iterable): List of camera numbers to be found in `folder`
        frames (iterable): Frame numbers to be read
        img_shape (list or tuple): Image shape in [width, height] format
        dark (np.ndarray, optional): Dark image to be subtracted from the averaged
            image. Defaults to None.

    Returns:
        np.ndarray: 3D Array of size (n_cameras, width, height) with the time-averaged
            image for each camera
    """
    img = np.zeros((len(cameras), *img_shape))
    for i, c in enumerate(cameras):
        img_folder = folder / f"camera {c}"
        img[i, :, :] = singlecam_mean(img_folder, frames, img_shape, dark)
    return img


def multicam_series(exp_folder, cameras, frames, img_shape, dark=None):
    """Read images from multiple cameras and return all frames

    Args:
        exp_folder (pathlib.Path): Path to the folder containing the `camera n` folders
        cameras (iterable): List of camera numbers to be found in `exp_folder`
        frames (iterable): Frame numbers to use
        img_shape (tuple[int, int]): Shape of the images to load.
        dark (np.ndarray or NoneType, optional): Dark image to subtract from
            final image. If None, no dark image subtraction is done. Defaults
            to None.

    Returns:
        np.ndarray: 4D array (n_frames, n_cams, img_height, img_width) of image data
    """
    scan_img = np.zeros((len(frames), len(cameras), *img_shape))
    for i, c in enumerate(cameras):
        img_folder = exp_folder / f"camera {c}"
        scan_img[:, i, :, :] = singlecam_series(img_folder, frames, img_shape, dark)
    return scan_img
