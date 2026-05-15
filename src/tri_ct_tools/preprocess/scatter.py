import asyncio
import sys
import numpy as np
import yaml
from pathlib import Path
import warnings

from tri_ct_tools.image.common import Image
from tri_ct_tools.image.reader import fetch_image_async, multicam_mean, singlecam_mean
from tri_ct_tools.image.writer import array_to_tif, write_tif_async


def write_mean_images(img, folder: Path, cams):
    """Write multi-camera image array to disk, one file per camera.

    Args:
        img (np.ndarray): 3D image array of shape (n_cameras, height, width).
        folder (Path): Output folder where camera subdirectories will be created.
        cams (list): List of camera numbers.
    """
    for i, cam in enumerate(cams):
        cam_img = img[i, ...]
        imfolder = folder / f"camera {cam}"
        array_to_tif(cam_img, imfolder, "average.tif")


async def process_image(in_queue: asyncio.Queue[Image], out_queue: asyncio.Queue[Image], scatter_img):
    """Asynchronously correct images by subtracting scatter contribution.

    Retrieves images from input queue, applies scatter correction using
    np.clip to ensure non-negative values, and places corrected images
    in output queue.

    Args:
        in_queue (asyncio.Queue[Image]): Queue of images to correct.
        out_queue (asyncio.Queue[Image]): Queue for corrected images.
        scatter_img (np.ndarray): Scatter reference image to subtract.
    """
    while True:
        img = await in_queue.get()
        if img is None:
            await out_queue.put(None)   # Signal for writing function to stop
            in_queue.task_done()
            break

        # print(f"correcting {img.name}")
        img.img = np.clip(img.img - scatter_img, 0, None)

        await out_queue.put(img)
        in_queue.task_done()


async def correct_cam_series(image_list: list[Image], scatter_img: np.ndarray):
    """Orchestrate asynchronous scatter correction for a camera's image series.

    Coordinates three concurrent tasks (fetch, process, write) using queues
    to efficiently pipeline image loading, scatter correction, and disk I/O.

    Args:
        image_list (list[Image]): List of Image objects for a single camera.
        scatter_img (np.ndarray): Scatter reference image for this camera.
    """
    process_queue = asyncio.Queue()
    write_queue = asyncio.Queue()

    fetch_task = asyncio.create_task(fetch_image_async(process_queue, image_list))
    process_task = asyncio.create_task(process_image(process_queue, write_queue, scatter_img))
    write_task = asyncio.create_task(write_tif_async(write_queue))

    await asyncio.gather(fetch_task, process_task)
    await process_queue.join()
    await write_queue.join()
    await write_task


def collect_images(exp_dir: Path, exp_out_folder: Path, cam):
    """Collect and organize image paths for a single camera from an experiment.

    Creates output directory structure and returns Image objects with
    input/output paths for all tif images from the specified camera.

    Args:
        exp_dir (Path): Root directory of the experiment.
        exp_out_folder (Path): Root output directory for corrected images.
        cam (int): Camera number to collect images for.

    Returns:
        list[Image]: Image objects with paths for loading and saving.

    Raises:
        AssertionError: If camera directory does not exist.
    """
    cam_dir = exp_dir / f"camera {cam}"
    assert cam_dir.exists(), f"Camera directory {cam_dir} does not exist."
    image_paths = cam_dir.glob("img_*.tif")

    cam_out_folder = exp_out_folder / f"camera {cam}"
    cam_out_folder.mkdir(parents=True, exist_ok=True)
    return [Image(path.name, path.absolute(), cam_out_folder / path.name) for path in image_paths]


def correct_series_async(scatter_images, exp_dir: Path, exp_out_folder: Path, cameras, frames, img_shape, offset):
    """Apply scatter correction to all cameras in an experiment's image series.

    Processes each camera sequentially, collecting images and applying
    asynchronous scatter correction using camera-specific scatter references.

    Args:
        scatter_images (np.ndarray): Array of scatter reference images, shape (n_cameras, height, width).
        exp_dir (Path): Root directory containing camera subdirectories.
        exp_out_folder (Path): Output directory for corrected images.
        cameras (list): List of camera numbers to process.
        frames (int): Number of frames (currently unused).
        img_shape (tuple): Expected image shape (height, width).
        offset: Offset parameter for output names (currently unused).
    """
    for cam in cameras:
        print(f"Correcting {exp_dir.name}, camera {cam}")
        image_list = collect_images(exp_dir, exp_out_folder, cam)
        cam_scatter = scatter_images[cam - 1, ...]
        asyncio.run(correct_cam_series(image_list, cam_scatter))
    print(f"Corrected {exp_dir.absolute()}")


def convert_name_to_scatter(exp_name, scatter_spec):
    """Convert experiment name to corresponding scatter measurement name.

    Inserts a scatter identifier into the experiment name for finding the
    corresponding scatter reference measurement.

    Args:
        exp_name (str): Original experiment name.
        scatter_spec (str): Scatter specification/identifier to insert.

    Returns:
        str: Modified experiment name with scatter specification.
    """
    scatter_part = f"Scatter_{scatter_spec}"
    exp_name_parts = exp_name.split("_")
    return f"{'_'.join(exp_name_parts[:-1])}_{scatter_part}_{exp_name_parts[-1]}"


def scatter_correct(yaml_file="inputs/scatter.yaml"):
    """Perform scatter correction on multi-camera X-ray images.

    Loads configuration from YAML file, processes all experiments in root folders,
    finds matching scatter measurements, and subtracts scatter from images.
    Saves corrected images to output folder.

    Args:
        yaml_file (str, optional): Path to scatter correction configuration file.
            Defaults to "inputs/scatter.yaml".

    Returns:
        None: Saves corrected images to disk.
    """
    # Load setup
    # Load calibration yaml with scatter scan properties
    with open(yaml_file) as scatter_yaml:
        settings = yaml.safe_load(scatter_yaml)

    root_folders = settings['roots']
    cameras = settings['cameras']
    average = settings['average']
    img_shape = (int(settings['img']['height']), int(settings['img']['width']))
    framerange = settings['frames']
    scatters = settings['scatter_IDs']
    # [ ] Scatters can be measured single-source or double-source. In the first
    # case, two scatter measurements need to be combined to obtain the camera-
    # specific scatter signal. In the first case, it comes from a single
    # measurement.

    # Correct for scatter
    # loop through the subdirectories in each root_folder
    for root in root_folders:
        rf = Path(root)
        if not rf.exists():
            warnings.warn(f"Provided path {rf} does not exist. Skipping.")
            continue
        output_folder = rf.parent / "03_scattercorrected"
        for subdir in rf.iterdir():
            if not subdir.is_dir():
                continue
            exp_name = subdir.name
            exp_out_folder = output_folder / exp_name

            if 'scatter' in exp_name.lower():
                continue

            n_missing = 0
            # [ ] add some output to a log file - if all are missing its a warning,
            # if one is missing an error. Log file should be marked with
            # datetimestamp.
            for i, sc_ID in enumerate(scatters):
                # [ ] Update the way scatter names are looked for. Current way
                # is very inflexible. Perhaps just look for the scatter ID
                # Double inflexible actually - should also create a method for
                # when single-source, multi-detector scatter measurements are
                # done. Those need addition of scatter values.
                scatter_name = convert_name_to_scatter(exp_name, scatters[i])
                if not Path(rf / scatter_name).exists():
                    n_missing += 1
            if n_missing:
                warnings.warn(f"{n_missing} scatter measurements missing "
                              f"for {scatter_name}. Skipping.", stacklevel=2)
                continue

            if "Empty" in exp_name:
                frange = framerange['empty']
                frames = range(frange['start'], frange['stop'], frange['step'])
            elif "Full" in exp_name:
                frange = framerange['full']
                frames = range(frange['start'], frange['stop'], frange['step'])
            elif "Dark" in exp_name:
                frange = framerange['dark']
                frames = range(frange['start'], frange['stop'], frange['step'])
            else:
                frange = framerange['measurement']
                frames = range(frange['start'], frange['stop'], frange['step'])

            scatter_images = np.zeros(shape=(len(cameras), *img_shape), dtype=np.int16)
            for i, sc_ID in enumerate(scatters):
                # Stick the right scatter identifier inbetween second-to-last and last
                scatter_name = convert_name_to_scatter(exp_name, sc_ID)
                scatter_dir = Path(rf / scatter_name)
                if not scatter_dir.exists():
                    warnings.warn(f"Could not find {scatter_dir.absolute()}.",
                                  stacklevel=2)
                    continue

                cam_folder = scatter_dir / f"camera {i+1}"
                # [ ] In the case of single source scatter, load this for each
                # source and add up.
                img = singlecam_mean(cam_folder, frames, img_shape).astype(np.int16)
                scatter_images[i, ...] = img

            if average:
                print("Correcting mean image...")
                images = multicam_mean(subdir, cameras, frames, img_shape).astype(np.int16)
                corrected_images = images - scatter_images
                write_mean_images(corrected_images, exp_out_folder, cameras)
            else:
                print("Correcting image series")
                # read image by image, correct and write to disk. asyncio might help here
                correct_series_async(scatter_images, subdir, exp_out_folder,
                                     cameras, frames, img_shape, frange['start'])


if __name__ == "__main__":
    # 2 ways to use:
    # 0. Provide no command line arguments. The YAML config in `inputs/` will be
    #   used for in- and outputs
    # 1. Provide the path to a config YAML in command line, this will be used.
    if len(sys.argv) > 1:
        config_path = Path(sys.argv[1])
        scatter_correct(config_path)
    else:
        scatter_correct()
