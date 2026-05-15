import asyncio

import numpy as np
from pathlib import Path
from matplotlib.figure import Figure
import tifffile

from tri_ct_tools.image.common import Image


def print_saving(file):
    print(f"Saving figure to {file}")


def fig_png(fig: Figure, output_folder: Path, filename: str, dpi=900):
    """Save a matplotlib figure as a PNG file.

    Args:
        fig (matplotlib.figure.Figure): Figure object to save.
        output_folder (Path): Directory where the file will be saved.
        filename (str): Filename (without extension).
        dpi (int, optional): Resolution in dots per inch. Defaults to 900.

    Returns:
        Path: Path to the saved PNG file.
    """
    output_folder.mkdir(parents=True, exist_ok=True)
    save_file = output_folder / filename
    save_file_png = save_file.with_suffix(".png")
    print_saving(save_file_png)
    fig.savefig(save_file_png, dpi=dpi)
    return save_file


def fig_svg(fig: Figure, output_folder: Path, filename: str):
    """Save a matplotlib figure as an SVG file.

    Args:
        fig (matplotlib.figure.Figure): Figure object to save.
        output_folder (Path): Directory where the file will be saved.
        filename (str): Filename (without extension).

    Returns:
        Path: Path to the saved SVG file.
    """
    output_folder.mkdir(parents=True, exist_ok=True)
    save_file = output_folder / filename
    save_file_svg = save_file.with_suffix(".svg")
    print_saving(save_file_svg)
    fig.savefig(save_file_svg, transparent=True)
    return save_file


def array_to_tif(img: np.ndarray, output_folder: Path, filename: str):
    """Save a numpy array as a TIFF image file.

    Args:
        img (np.ndarray): Image array to save.
        output_folder (Path): Directory where the file will be saved.
        filename (str): Filename (without extension).

    Returns:
        Path: Path to the saved TIFF file.
    """
    output_folder.mkdir(parents=True, exist_ok=True)
    output_file = output_folder / filename
    output_tif = output_file.with_suffix(".tif")
    tifffile.imwrite(output_tif, img.astype(np.int16))
    return output_tif


async def write_tif_async(queue: asyncio.Queue[Image]):
    """Asynchronously write corrected images to disk.

    Retrieves processed images from queue and writes them as 16-bit
    TIFF files to the output paths specified in Image objects.

    Args:
        queue (asyncio.Queue[Image]): Queue of corrected images to write.
    """
    while True:
        img = await queue.get()
        if img is None:
            queue.task_done()
            break

        # print(f"writing {img.name}")
        await asyncio.to_thread(tifffile.imwrite, img.out_path, img.img.astype(np.int16))
        queue.task_done()
