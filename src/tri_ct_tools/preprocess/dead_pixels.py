import os
import sys
import filecmp
import yaml
import numpy as np
from datetime import datetime
from PIL import Image
from pathlib import Path

from tri_ct_tools.image.reader import single, singlecam_mean
from tri_ct_tools.image.writer import array_to_tif


def read_detector_settings(source_dir: Path):
    filename = source_dir / "settings  data.txt"

    cam_mode = ''
    VROI = [0, 1]
    framerate = 0
    timestamp = datetime.now()
    cameras = []

    settings = {}
    with open(filename) as f:
        for line in f:
            split_line = line.split(":")

            # TriggerDelay does not contain the colon, and would break the rest of the flow.
            if len(split_line) < 2:
                continue
            (key, val) = split_line

            val = val.strip()
            settings[key] = val

    cam_mode = str(settings['Cam Mode'])
    if cam_mode == "MAG0":
        VROI = [0, 1523]

    if cam_mode == "VROI":
        VROI[0] = int(settings['StartLine1'])
        VROI[1] = int(settings['StopLine1'])

    framerate = int(settings['Framerate'])
    timestamp = datetime.strptime(settings['Date/time'], '%Y%m%d_%H%M%S_0')

    for i in range(1, 4):
        if settings[f'Cam {i}'] == "On":
            cameras.append(i)

    return (cam_mode, VROI, framerate, timestamp, cameras)


def vertical_interpolation(image, x, y):
    # dead pixel correction through vertical interpolation
    # interpolate between neighbours, round to nearest integer

    image[y, x] = np.round((image[y - 1, x] + image[y + 1, x]) / 2)

    return image


def vertical_interpolation_vec(image, x_min, x_max, y):
    # dead pixel correction through vertical interpolation
    # interpolate between neighbours, round to nearest integer

    x_range = slice(x_min, x_max+1)
    return np.round((image[y - 1, x_range] + image[y + 1, x_range]) / 2)


def double_horizontal_interpolation(image, x1, x2, y):
    # dead pixel correction through horizontal interpolation (case of 2 dead lines)
    # interpolate between neighbours, round to nearest integer

    image[y, x1] = np.round((image[y, x1 - 1] + image[y, x2 + 1]) / 2)
    image[y, x2] = image[y, x1]
    return image


def double_horizontal_interpolation_vec(image, x1, x2):
    # dead pixel correction through horizontal interpolation (case of 2 dead lines)
    # interpolate between neighbours, round to nearest integer

    n_tiles = abs(x1-x2) + 1
    interpolated_values = np.round((image[:, x1 - 1] + image[:, x2 + 1]) / 2)
    return np.tile(interpolated_values, (n_tiles, 1)).transpose()


def horizontal_interpolation(image, x, y):
    # dead pixel correction through horizontal interpolation
    # interpolate between neighbours, round to nearest integer

    image[y, x] = round((image[y, x - 1] + image[y, x + 1]) / 2)

    return image


def horizontal_interpolation_vec(image, x, y_min, y_max):
    # dead pixel correction through horizontal interpolation
    # interpolate between neighbours, round to nearest integer

    y_range = slice(y_min, y_max+1)
    return np.round((image[y_range, x - 1] + image[y_range, x + 1]) / 2)


def correct_lines(image, VROI, hline, vert_tophalf, vert_bothalf, double_vert):
    y_max = image.shape[0]
    x_max = image.shape[1]

    hline_included = True

    if hline > 0:
        image[0:hline+1, vert_tophalf] = horizontal_interpolation_vec(image, vert_tophalf, 0, hline)
    else:
        hline_included = False

    if hline < (VROI[1] - VROI[0]):
        image[hline:y_max, vert_bothalf] = horizontal_interpolation_vec(image, vert_bothalf, hline, y_max)
    else:
        hline_included = False

    if hline_included:
        image[hline, :] = vertical_interpolation_vec(image, 0, x_max, hline)

    # horizontal line at the very top of detector
    image[0, :] = image[1, :]

    x1 = min(double_vert)
    x2 = max(double_vert)
    image[:, x1:x2+1] = double_horizontal_interpolation_vec(image, x1, x2)

    return image


def dead_pixel_correction(image, cam_no, offsets, VROI=[0, 1523]):
    """Take an image and correct the dead pixels

    The function currently only corrects for dead pixel lines, not for individual
    dead pixels. Identification and correction of these could still be done in the
    future.

    Args:
        image (np.ndarray): 2D image with dead pixels
        cam_no (int): camera number
        offsets (dict): Dictionary with the horizontal and vertical offsets for this camera
        VROI (list, optional): VROI settings for the measurement. Defaults to [0, 1523].

    Returns:
        np.ndarray: 2D image with dead pixels corrected through interpolation
    """
    # Dead pixel (-lines) correction
    # pixels are defined based on camera 1 (*73), need to check for other cameras!
    # especially for single pixel corrections, which are different for all cameras.

    # define ROI used, to correct the line numbers with
    start_line_correction = 12

    if min(VROI) == 0:
        start_line = 0
    else:
        # [ ] Check what happens when VROI is set top the top part of the detector
        # Start line would still be 0, but is the offset needed? i.e. is the offset
        # VROI-dependent or start line location dependent? And what happens when
        # the start line is < 12 here?
        # if not zero, requires addition of 12. Not sure yet why this is needed, or if its the same every time
        start_line = min(VROI) - start_line_correction

    # Correct hline position for VROI settings
    hline = 763 - start_line

    with open('src/tri_ct_tools/preprocess/dead_pixel_lines.yaml') as dpl_file:
        dpl = yaml.safe_load(dpl_file)

    vert_tophalf = offsets[cam_no]['vertical']['top'] + np.array(dpl[f'cam{cam_no}']['vertical']['top'])
    vert_bothalf = offsets[cam_no]['vertical']['bot'] + np.array(dpl[f'cam{cam_no}']['vertical']['bot'])

    double_vert = np.array(dpl[f'cam{cam_no}']['double_vert']) + offsets[cam_no]['double_vert']

    image = correct_lines(image, VROI, hline, vert_tophalf, vert_bothalf, double_vert)

    return image


def find_subdirectories(directory: Path, subdirectories=None) -> list[Path]:
    
    if subdirectories is None:
        subdirectories = []

    for entry in directory.iterdir():
        if entry.is_dir():
            subdirectories.append(entry)

    return subdirectories


def list_subdirectories(source_dir: str | list, target_dir: str | list) -> tuple[list, list]:
    if isinstance(source_dir, str) and isinstance(target_dir, str):
        # Simple case: find subdirectories and return a list of those, with same
        # target dir for each
        subdirs = find_subdirectories(Path(source_dir))
        target_dirs = [Path(target_dir)] * len(subdirs)
        return subdirs, target_dirs

    if not isinstance(source_dir, list) and isinstance(target_dir, list):
        raise ValueError(
            "Unexpected input types for list_subdirectories(). Arguments should "
            "be either both strings or both lists"
        )
    # get subdirs for each entry in source_dir
    source_dirs = []
    target_dirs = []
    for sdir, tdir in zip(source_dir, target_dir):
        subdirs = find_subdirectories(Path(sdir))
        source_dirs += subdirs
        target_dirs += [Path(tdir)] * len(subdirs)

    return source_dirs, target_dirs


def process_file(i, file: Path, n_cam, VROI, target_dir, total_files, offsets, dark_img):
    output_file = target_dir / file.name

    if output_file.exists() and filecmp.cmp(file, output_file):
        print(f"\nFile {file.name} already exists, skipping")
    else:
        print(f"Frame number {i + 1} / {total_files}", end='\r', flush=True)
        image_array = single(file, quiet=True)
        np.clip(image_array - dark_img, 0, None, out=image_array)

        image_array = dead_pixel_correction(image_array, n_cam, offsets, VROI=VROI)

        array_to_tif(image_array, target_dir, file.name)
        # Image.fromarray(image_array).save(output_file)


def load_darks(config):
    dirs = config['dark']
    darkframes = range(
        config['darkframes']['start'],
        config['darkframes']['stop'],
        config['darkframes']['step']
    )
    # Shortcut for when a single dark image is provided.
    if isinstance(dirs, str):
        dark_path = Path(dirs)
        cam_folders = list(dark_path.glob("camera*"))
        img_shape = single(cam_folders[0] / 'img_10.tif').shape
        dark_img = np.zeros((len(cam_folders), *img_shape), dtype=np.int16)
        for i, cam_folder in enumerate(cam_folders):
            assert cam_folder.is_dir()
            dark_img[i, ...] = singlecam_mean(cam_folder, darkframes, img_shape)
        return dark_img

    darks = {}
    dark_info = [
        [],     # crop
        [],     # startline
        [],     # stopline
        [],     # image size
        [],     # framerate
        [],     # date
        [],     # cameras
        [],     # images
    ]
    for dir in dirs:
        dark_path = Path(dir)
        cam_folders = list(dark_path.glob("camera*"))
        img_shape = single(cam_folders[0] / 'img_10.tif').shape
        # load images
        dark_img = np.zeros((len(cam_folders), *img_shape), dtype=np.int16)
        for i, cam_folder in enumerate(cam_folders):
            assert cam_folder.is_dir()
            dark_img[i, ...] = singlecam_mean(cam_folder, darkframes, img_shape)
        # Obtain metadata
        xray_settings = read_detector_settings(dark_path)
        (cam_mode, VROI, framerate, timestamp, cameras) = xray_settings
        dark_info[0].append(cam_mode)
        dark_info[1].append(VROI[0])
        dark_info[2].append(VROI[1])
        dark_info[3].append(img_shape)
        dark_info[4].append(framerate)
        dark_info[5].append(timestamp)
        dark_info[6].append(cameras)
        dark_info[7].append(dark_img)

    # Convert dark info to dictionary for more meaningful indexing
    darks['crop'] = np.array(dark_info[0])
    darks['startline'] = np.array(dark_info[1])
    darks['stopline'] = np.array(dark_info[2])
    darks['img_shape'] = np.array(dark_info[3])
    darks['framerate'] = np.array(dark_info[4])
    darks['timestamp'] = np.array(dark_info[5])
    darks['cameras'] = dark_info[6]
    # images will have shape (n_darks, n_cams, *img_shape)
    darks['images'] = dark_info[7]
    return darks


def pick_dark(source_dir, darks):
    if isinstance(darks, np.ndarray):
        return darks
    source_settings = read_detector_settings(source_dir)
    cam_folders = list(source_dir.glob("camera*"))
    img_shape = single(cam_folders[0] / 'img_10.tif').shape

    (cam_mode, VROI, framerate, timestamp, cameras) = source_settings
    
    same_shape = (darks['img_shape'] == img_shape).all(axis=1)
    same_framerate = darks['framerate'] == framerate
    full_shape = (darks['img_shape'] == (1524, 1548)).all(axis=1)
    if np.sum(same_shape & same_framerate) == 1:
        dark_img = darks['images'][np.argmax(same_shape & same_framerate)]
    elif np.sum(same_shape & same_framerate) > 1:
        # Pick the closest date
        timedeltas = abs(darks['timestamp'] - timestamp)
        valid_idx = np.where(same_shape & same_framerate)[0]
        idx_date_based = valid_idx[timedeltas[valid_idx].argmin()]
        # idx_date_based = np.argmin(abs(darks['timestamp'][same_shape & same_framerate] - timestamp))
        dark_img = darks['images'][idx_date_based]
    elif np.sum(same_framerate & full_shape) >= 1:
        # Find the shapes that are larger or equal to shape of source
        dark_img = darks['images'][np.argmax(full_shape & same_framerate)]
        # Crop to source image size
        dark_img = dark_img[:, VROI[0]:VROI[1], :]
    else:
        raise LookupError(f"Could not find dark image for {source_dir}. No dark "
                          "images with the same framerate and equal or larger "
                          "image size provided.")

    return dark_img


def main(root_source_dir, root_target_dir):
    # initial processing of raw data (dead pixel correction, rotate, flip, contrast)
    with open('D:\XRay\XRay Summer 2025 - Reanalysis\dead_pixel_reanalysis.yaml') as dp_file:
        config = yaml.safe_load(dp_file)
    if not root_source_dir:
        root_source_dir = config['input_folder']
        root_target_dir = config['output_folder']

    copy_raw = bool(config['copy_raw'])
    avg_only = bool(config['avg_only'])
    if avg_only:
        framestart = int(config['framestart'])
    else:
        framestart = None
    offsets = config['offsets']
    # VROI = get_VROI_setting()

    # Get a list of all subfolders in this folder.
    source_dirs, target_dirs = list_subdirectories(root_source_dir, root_target_dir)
    darks = load_darks(config)

    for source_dir, target_dir in zip(source_dirs, target_dirs):
        # skip already preprocessed directories
        if 'preprocessed' in source_dir.name:
            continue

        dark_images = pick_dark(source_dir, darks)

        # Make output directory
        preprocessed_dir = target_dir / source_dir.name
        os.makedirs(preprocessed_dir, exist_ok=True)

        if copy_raw:
            raw_dir = target_dir.parent / '01_raw' / source_dir.name
            raw_dir.mkdir(parents=True, exist_ok=True)

        # gather settings from "setting  data.txt"
        xray_settings = read_detector_settings(source_dir)
        VROI = xray_settings[1]

        # Copy "settings  data.txt" files for camera folder
        os.system(f'robocopy "{source_dir}" "{preprocessed_dir}" '
                  f'"settings  data.txt" /njh /njs /ndl /nc /ns')
        if copy_raw:
            os.system(f'robocopy "{source_dir}" "{raw_dir}" '
                      f'"settings  data.txt" /njh /njs /ndl /nc /ns')

        for i, n_cam in enumerate(range(1, 4)):
            dark_img = dark_images[i, :, :]
            camdir = source_dir / f'camera {str(n_cam)}'
            # load filenames of all images in directory
            total_files = sum(1 for _ in camdir.glob('img_*.tif'))

            if total_files == 0:
                print(f"No files in {camdir}. That might be a problem.")
                continue

            # Create output directory
            output_directory = preprocessed_dir / f"camera {n_cam}"
            output_directory.mkdir(parents=True, exist_ok=True)

            if copy_raw:
                raw_output_dir = raw_dir / f"camera {n_cam}"
                raw_output_dir.mkdir(parents=True, exist_ok=True)

            # Copy "timestamp data.txt" files for camera folder
            os.system(f'robocopy "{camdir}" "{output_directory}" '
                      f'"timestamp data.txt"  /njh /njs /ndl /nc /ns')
            if copy_raw:
                os.system(f'robocopy "{camdir}" "{raw_output_dir}" '
                          f'"timestamp data.txt"  /njh /njs /ndl /nc /ns')

            print(f"Processing files in {camdir}")
            img_list = list(camdir.glob("img_*.tif"))
            img_shape = single(img_list[0], quiet=True).shape
            if avg_only:
                # Determine framerange from number of 'img_*.tif' files.
                n_frames = len(img_list)
                framerange = range(framestart, n_frames)
                # read mean of img files in the folder
                img_array = singlecam_mean(camdir, framerange, img_shape,
                                           dark=dark_img, quiet=True)
                # apply dpc to mean img
                img_array = dead_pixel_correction(img_array, n_cam, offsets,
                                                  VROI)
                # write that to target_dir / average.tif
                array_to_tif(img_array, output_directory, 'average.tif')
            else:
                for j, file in enumerate(img_list):
                    process_file(j, file, n_cam, VROI, output_directory,
                                 total_files, offsets, dark_img)

            if copy_raw:
                print('Copying raw images...')
                for j, file in enumerate(img_list):
                    os.system(f'robocopy "{camdir}" "{raw_output_dir}" '
                              f'"{file.name}"  /njh /njs /ndl /nc /ns')
            print('')


if __name__ == "__main__":
    # 4 ways to use:
    # 0. Provide no command line arguments. You will be prompted for a source
    #   directory. For the rest of the behaviour, see 1
    # 1. Provide only source directory. Expected to have a `raw` folder in there,
    #   with the folders with pictures. Output will be written to
    #   `source / preprocessed`.
    # 2. Provide both source and target. Source is expected to immediately have
    #   experiment folders in there. Output will be written to the target
    #   directory
    # 3. Provide source, target and a copy-raw flag. If the copy-raw flag is
    #   True, copy the raw files to `target / raw`.
    if len(sys.argv) > 1:
        source_dir = Path(sys.argv[1])
    else:
        source_dir = False

    if len(sys.argv) > 2:
        target_dir = Path(sys.argv[2])
    else:
        target_dir = False

    main(source_dir, target_dir)
