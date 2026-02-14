from matplotlib.animation import FuncAnimation
import pathlib
import matplotlib.pyplot as plt
import numpy as np

from tri_ct_tools.colors import blues9_map
from tri_ct_tools.image.writer import print_saving


def create_animation(
        output_folder: pathlib.Path,
        cameras: list,
        image_series: np.ndarray,
        filename: str | None = None,
        frame_start: int = 0,
        framerate: int = 22,
        fl: float | None = None,
        fcounter: int = 0):
    """Create and save an animation from a series of multi-camera images.

    Creates a matplotlib animation displaying images from multiple cameras side-by-side
    with a time counter. The animation is saved as an AVI file.

    Args:
        output_folder (pathlib.Path): Directory where the animation file will be saved.
        cameras (list): List of camera numbers to display.
        image_series (np.ndarray): 4D array of shape (n_frames, n_cameras, height, width).
        filename (str | None, optional): Name of the output file. If None, a name is
            generated using fcounter and other parameters. Defaults to None.
        frame_start (int, optional): Frame number offset for time display. Defaults to 0.
        framerate (int, optional): Framerate of the animation in frames per second.
            Defaults to 22.
        fl (float | None, optional): Flow rate parameter used in filename generation.
            Defaults to None.
        fcounter (int, optional): Frame counter for filename generation. Defaults to 0.

    Returns:
        int: Updated fcounter value (fcounter + 1).
    """
    # First dimension of image series is the number of frames
    n_frames = image_series.shape[0]
    fig, axs = plt.subplots(1, len(cameras), figsize=(10, 4), layout="constrained")
    current_images = []
    for i, c in enumerate(cameras):
        current_images.append(axs[i].imshow(image_series[0, i, :, :],
                                            aspect='equal',
                                            cmap=blues9_map,
                                            vmin=0,
                                            vmax=.5))
        axs[i].axis('off')
        axs[i].set_title(f"Camera {c}")
    stitle = fig.suptitle(f"Gas fraction @ {frame_start/framerate:>6.2f} s")

    fig.colorbar(current_images[-1], ax=axs, orientation='vertical', fraction=.1)

    def update_image(fr):
        for i, c in enumerate(cameras):
            current_images[i].set(data=image_series[fr, i, :, :])
        stitle.set_text(f"Gas fraction @ {(frame_start + fr)/framerate:>6.2f} s")
        return [*current_images, stitle]

    # frame-to-frame interval in ms
    interval_ms = 1 / framerate * 1000
    ani = FuncAnimation(fig, update_image, range(1, n_frames), interval=interval_ms)
    if filename is None and fl is not None and fcounter is not None:
        filename = (f"{fcounter:02n}_cam{'-'.join(map(str, cameras))}_holdup_water"
                    f"_{fl}_movie_{n_frames}-frames_cam-{cameras}.avi"
                    )
    fcounter += 1
    save_file_avi = output_folder / filename
    print_saving(save_file_avi)
    ani.save(save_file_avi,
             fps=framerate,
             dpi=300,
             progress_callback=lambda i, n: print(f'Saving frame {i + 1}/{n}'))
    return fcounter
