import numpy as np
from pathlib import Path
from matplotlib.figure import Figure
import tifffile


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
