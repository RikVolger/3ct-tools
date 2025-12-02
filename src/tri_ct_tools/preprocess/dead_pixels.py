import os
import sys
import filecmp
import yaml
import numpy as np
from PIL import Image
from pathlib import Path

from tri_ct_tools.image.writer import array_to_tif


def read_detector_settings(source_dir):
    filename = Path(source_dir) / "settings  data.txt"

    cam_mode = False
    VROI = [0, 1]

    with open(filename) as f:
        for line in f:
            split_line = line.split(":")

            # TriggerDelay does not contain the colon, and would break the rest of the flow.
            if len(split_line) < 2:
                continue
            else:
                (key, val) = split_line

            val = val.strip()

            if key == "Cam Mode":
                cam_mode = val
                if cam_mode == "MAG0":
                    VROI = [0, 1523]
                    break

            if cam_mode == "VROI" and key == "StartLine1":
                VROI[0] = int(val)

            if cam_mode == "VROI" and key == "StopLine1":
                VROI[1] = int(val)

    return (cam_mode, VROI)


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


def find_subdirectories(directory: Path, subdirectories=[]) -> list[Path]:
    for entry in directory.iterdir():
        if entry.is_dir():
            subdirectories.append(entry)

    return subdirectories


def process_file(i, file: Path, n_cam, VROI, target_dir, total_files, offsets):
    output_file = target_dir / file.name

    if output_file.exists() and filecmp.cmp(file, output_file):
        print(f"File {file.name} already exists, skipping")
    else:
        print(f"Frame number {i + 1} / {total_files}")
        # [ ] Would like to use reader functionalities here, but would overload
        # the outputs. Maybe create with a printing flag?
        image_array = np.array(Image.open(file))

        image_array = dead_pixel_correction(image_array, n_cam, offsets, VROI=VROI)

        array_to_tif(image_array, target_dir, file.name)
        # Image.fromarray(image_array).save(output_file)


def main(source_dir, target_dir):
    # initial processing of raw data (dead pixel correction, rotate, flip, contrast)
    with open('inputs/dead_pixel.yaml') as dp_file:
        config = yaml.safe_load(dp_file)
    if not source_dir:
        source_dir = config['input_folder']
        target_dir = Path(config['output_folder'])

    copy_raw = config['copy_raw']
    offsets = config['offsets']
    # VROI = get_VROI_setting()

    # Get a list of all subfolders in this folder.
    subdirectories = find_subdirectories(source_dir)

    for subdir in subdirectories:
        # skip already preprocessed directories
        if 'preprocessed' in subdir.name:
            continue

        # Make output directory
        preprocessed_dir = target_dir / subdir.name
        os.makedirs(preprocessed_dir, exist_ok=True)

        if copy_raw:
            raw_dir = target_dir.parent / '01_raw' / subdir.name
            raw_dir.mkdir(parents=True, exist_ok=True)

        # gather settings from "setting  data.txt"
        (cam_mode, VROI) = read_detector_settings(subdir)

        # Copy "settings  data.txt" files for camera folder
        os.system(f'robocopy "{subdir}" "{preprocessed_dir}" "settings  data.txt" /njh /njs /ndl /nc /ns')
        if copy_raw:
            os.system(f'robocopy "{subdir}" "{raw_dir}" "settings  data.txt" /njh /njs /ndl /nc /ns')

        for n_cam in range(1, 4):
            camdir = subdir / f'camera {str(n_cam)}'
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
            os.system(f'robocopy "{camdir}" "{output_directory}" "timestamp data.txt"  /njh /njs /ndl /nc /ns')
            if copy_raw:
                os.system(f'robocopy "{camdir}" "{raw_output_dir}" "timestamp data.txt"  /njh /njs /ndl /nc /ns')

            for i, file in enumerate(camdir.glob("img_*.tif")):
                process_file(i, file, n_cam, VROI, output_directory, total_files, offsets)
                if copy_raw:
                    os.system(f'robocopy "{camdir}" "{raw_output_dir}" "{file.name}"  /njh /njs /ndl /nc /ns')


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
