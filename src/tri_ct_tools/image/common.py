from pathlib import Path
import numpy as np


class Image(object):
    """Container for image data and metadata during processing.

    Attributes:
        name (str): Name identifier for the image.
        path (Path): File path to the image.
        out_path (Path): Output file path for the corrected image.
        img (np.ndarray, optional): Image array data, initially None.
    """
    def __init__(self, name, path, out_path):
        super(Image, self).__init__()
        self.name: str = name
        self.path: Path = path
        self.out_path: Path = out_path
        self.img: None | np.ndarray = None
