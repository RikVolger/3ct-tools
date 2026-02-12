import numpy as np
import yaml
from pathlib import Path
import warnings

from tri_ct_tools.image.reader import multicam_mean, singlecam_mean
from tri_ct_tools.image.writer import array_to_tif


def write_images(img, folder: Path, cams):
    for i, cam in enumerate(cams):
        cam_img = img[i, ...]
        imfolder = folder / f"camera {cam}"
        array_to_tif(cam_img, imfolder, "average.tif")


def convert_name_to_scatter(exp_name, scatter_spec):
    scatter_part = f"Scatter_{scatter_spec}"
    exp_name_parts = exp_name.split("_")
    return f"{'_'.join(exp_name_parts[:-1])}_{scatter_part}_{exp_name_parts[-1]}"


def scatter_correct(yaml_file="inputs/scatter.yaml"):
    # Load setup
    # Load calibration yaml with scatter scan properties
    with open(yaml_file) as scatter_yaml:
        settings = yaml.safe_load(scatter_yaml)

    root_folders = settings['roots']
    cameras = settings['cameras']
    average = settings['average']
    img_shape = (int(settings['img']['height']), int(settings['img']['width']))
    if average:
        framerange = settings['frames']
    else:
        raise ValueError("Scatter correction for time-resolved images is"
                         "currently undefined. Expect errors.")
    scatters = settings['scatter_IDs']

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

            if 'Scatter' in exp_name or 'scatter' in exp_name:
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

            images = multicam_mean(subdir, cameras, frames, img_shape).astype(np.int16)
            scatter_images = np.zeros_like(images)
            for i, sc_ID in enumerate(scatters):
                # Stick the right scatter identifier inbetween second-to-last and last
                scatter_name = convert_name_to_scatter(exp_name, scatters[i])
                scatter_dir = Path(rf / scatter_name)
                if not scatter_dir.exists():
                    warnings.warn(f"Could not find {scatter_dir.absolute()}.",
                                  stacklevel=2)
                    continue

                cam_folder = scatter_dir / f"camera {i+1}"
                img = singlecam_mean(cam_folder, frames, img_shape).astype(np.int16)
                scatter_images[i, ...] = img

            print("Correcting image...")
            corrected_images = images - scatter_images

            exp_out_folder = output_folder / exp_name

            write_images(corrected_images, exp_out_folder, cameras)


if __name__ == "__main__":
    scatter_correct(R"D:\XRT paper\XRay\scatter.yaml")
