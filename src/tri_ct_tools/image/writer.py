from pathlib import Path
from matplotlib.figure import Figure


def print_saving(file):
    print(f"Saving figure to {file}")


def save_png(fig: Figure, output_folder: Path, filename: str, dpi=900):
    save_file = output_folder / filename
    save_file_png = save_file.with_suffix(".png")
    print_saving(save_file_png)
    fig.savefig(save_file_png, dpi=dpi)


def save_svg(fig: Figure, output_folder: Path, filename: str):
    save_file = output_folder / filename
    save_file_svg = save_file.with_suffix(".svg")
    print_saving(save_file_svg)
    fig.savefig(save_file_svg, transparent=True)
