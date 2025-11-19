import numpy as np
from pathlib import Path
from matplotlib.figure import Figure
import tifffile


def print_saving(file):
    print(f"Saving figure to {file}")


def fig_png(fig: Figure, output_folder: Path, filename: str, dpi=900):
    output_folder.mkdir(parents=True, exist_ok=True)
    save_file = output_folder / filename
    save_file_png = save_file.with_suffix(".png")
    print_saving(save_file_png)
    fig.savefig(save_file_png, dpi=dpi)
    return save_file


def fig_svg(fig: Figure, output_folder: Path, filename: str):
    output_folder.mkdir(parents=True, exist_ok=True)
    save_file = output_folder / filename
    save_file_svg = save_file.with_suffix(".svg")
    print_saving(save_file_svg)
    fig.savefig(save_file_svg, transparent=True)
    return save_file


def array_to_tif(img: np.ndarray, output_folder: Path, filename: str):
    output_folder.mkdir(parents=True, exist_ok=True)
    output_file = output_folder / filename
    output_tif = output_file.with_suffix(".tif")
    tifffile.imwrite(output_tif, img.astype(np.int32))
    return output_tif
