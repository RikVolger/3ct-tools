from typing import Iterable

from matplotlib.animation import FuncAnimation
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mplc
from matplotlib.colors import ListedColormap
import numpy as np
import pyvista as pv

from tri_ct_tools.colors import blues9_map
from tri_ct_tools.image.writer import print_saving


def generate_colormap(colormap):
    if colormap is None:
        colormap = blues9_map
    elif isinstance(colormap, str):
        colormap = mpl.colormaps[colormap]
    elif isinstance(colormap, mplc.Colormap):
        colormap = colormap
    else:
        raise ValueError(f"Input colormap ({type(colormap)}) should be None, "
                         f"str or an instance of matplotlib.colors.Colormap.")

    return colormap


def read_vtk(file: Path):
    """Read a .vtk file of reconstruction data from disk

    Args:
        file (Path): The path (pathlib.Path) to the file to read.

    Returns:
        tuple[np.ndarray, dict]: First return value is the 3D np.ndarray of 
            gas fraction values. Second return value is a user dictionary 
            of reconstruction metadata.
    """
    mesh = pv.read(file, progress_bar=True)

    density_data = mesh.get_array("Density")
    point_data_3d = np.array(density_data.reshape(mesh.dimensions, order='F'))

    mesh_metadata = mesh.user_dict

    return point_data_3d, mesh_metadata


def midpoints(x):
    sl = ()
    for _ in range(x.ndim):
        x = (x[sl + np.index_exp[:-1]] + x[sl + np.index_exp[1:]]) / 2.0
        sl += np.index_exp[:]
    return x


def create_3d_animation(
        source_folder: Path,
        filenames: Iterable[str],
        output_folder: Path,
        output_name: Path = None,
        colormap: mplc.Colormap | str = "viridis",
        frame_start: int = 0,
        framerate: int = 22,
        flowrate: float | None = None,
        min_holdup: float = 0.3,
        max_holdup: float = 0.5,
        column_diameter: float = 19.2,
        ):
    colormap = generate_colormap(colormap)
    n_frames = len(filenames)
    
    print(f"Reading {source_folder / filenames[0]}")
    initial_holdup, metadata = read_vtk(source_folder / filenames[0])

    # Volume side (cm) divided by voxel size (cm) gives number of voxels for x/y
    xy_center = metadata['volume_side'] / 2
    # Create a cleaning filter
    # Cut off the top and bottom 10 pixels
    # Mask to the column cylinder - set outside pixels to (0 or -1)
    
    # Create 3D image with holdup data.

    # drawing volume coordinates one bigger than data volume because thats 
    # what ax.voxels() wants.
    print("Creating volume indices")
    xv, yv, zv = np.indices((s + 1 for s in initial_holdup.shape)) * metadata['voxel_size']
    
    print("Making data selection")
    # Column mask should have each dimension smaller by one. Use these for that.
    xc, yc, zc = np.indices(initial_holdup.shape) * metadata['voxel_size']
    print(f"Initial points: {np.prod(initial_holdup.shape)}")
    # Only select voxels inside the column
    column = ((xc - xy_center)**2 + (yc - xy_center)**2) < (column_diameter / 2)**2
    print(f"Points inside column: {np.sum(column.astype(bool))}")
    # Only use a quarter of the column
    part = ((xc - xy_center) > 0) & ((yc - xy_center) > 0)
    print(f"Points inside the quarter: {np.sum(part.astype(bool))}")
    # Crop top and bottom parts off
    bottom_lim = zc > np.floor(initial_holdup.shape[2] * metadata['voxel_size'] * 0.1)
    print(f"Points above bottom limit: {np.sum(bottom_lim.astype(bool))}")
    top_lim = zc < np.floor(initial_holdup.shape[2] * metadata['voxel_size'] * 0.9)
    print(f"Points below top limit: {np.sum(top_lim.astype(bool))}")
    # only show the parts above minimum holdup
    gas = initial_holdup > min_holdup
    print(f"Voxels with enough gas: {np.sum(gas.astype(bool))}")

    selection = column & part & bottom_lim & top_lim & gas
    print(f"Final selection: {np.sum(selection.astype(bool))}")

    print("Creating color arrays")
    normalized_holdup_for_rgb = np.clip(
        (initial_holdup - min_holdup) / (max_holdup - min_holdup),
        0,
        1
    )
    # set the colors of each object
    # Get RGB from colormap
    rgb = colormap(normalized_holdup_for_rgb)[..., :3]

    normalized_holdup_for_a = np.clip(
        (initial_holdup - min_holdup) / (max_holdup - min_holdup),
        0.2,
        0.6)
    # Stack with a-normalized values for alpha value
    rgba = np.concatenate([
        rgb,
        normalized_holdup_for_a[..., np.newaxis]
    ], axis=-1)

    # and plot everything
    print("Generating figure")
    ax = plt.figure().add_subplot(projection='3d', aspect='equal')
    print("Adding voxels...")
    ax.voxels(xv, yv, zv, selection, facecolors=rgba, edgecolors="#00000000")

    plt.show()
    ## ABANDONED
    # Seemed to take too long so looked for alternatives. Came to pyvista.
    # Revisit? Pyvista is also taking long when you properly plot volumes...
    ###
    # Opacity (alpha) channel based on holdup
    # Also colormap (default Viridis) based on holdup
    # Set camera position to slightly above the top of the volume and 3 column 
    # lengths away, looking at the center of the column
    # Store camera for access in animating (want to make it possible to move)
    # fig, axs = plt.subplots(1, len(cameras), figsize=(10, 4), layout="constrained")
    # current_images = []
    # for i, c in enumerate(cameras):
    #     current_images.append(axs[i].imshow(image_series[0, i, :, :],
    #                                         aspect='equal',
    #                                         cmap=colormap,
    #                                         vmin=0,
    #                                         vmax=.5))
    #     axs[i].axis('off')
    #     axs[i].set_title(f"Camera {c}")
    # stitle = fig.suptitle(f"Gas fraction @ {frame_start/framerate:>6.2f} s")

    # fig.colorbar(current_images[-1], ax=axs, orientation='vertical', fraction=.1)

    # def update_image(fr):
    #     for i, c in enumerate(cameras):
    #         current_images[i].set(data=image_series[fr, i, :, :])
    #     stitle.set_text(f"Gas fraction @ {(frame_start + fr)/framerate:>6.2f} s")
    #     return [*current_images, stitle]

    # # frame-to-frame interval in ms
    # interval_ms = 1 / framerate * 1000
    # ani = FuncAnimation(fig, update_image, range(1, n_frames), interval=interval_ms)
    # if filename is None and fl is not None and fcounter is not None:
    #     filename = (f"{fcounter:02n}_cam{'-'.join(map(str, cameras))}_holdup_water"
    #                 f"_{fl}_movie_{n_frames}-frames_cam-{cameras}.avi"
    #                 )
    # fcounter += 1
    # save_file_avi = output_folder / filename
    # print_saving(save_file_avi)
    # ani.save(save_file_avi,
    #          fps=framerate,
    #          dpi=300,
    #          progress_callback=lambda i, n: print(f'Saving frame {i + 1}/{n}'))
    # return fcounter


def time_string(framerate: int | float, current_frame: int | float) -> str:
    return f"Gas fraction @ {current_frame / framerate:>6.2f} s"


def create_3d_volume_animation_pv(
        filenames: Iterable[Path],
        output_folder: Path,
        output_name: str,
        colormap: mplc.Colormap | str = "viridis",
        framerate: int = 22,
        flowrate: float | None = None,
        min_holdup: float = 0.3,
        max_holdup: float = 0.5,
        column_diameter: float = 19.2,
        ):
    colormap = generate_colormap(colormap)

    volume_series = pv.read(filenames)
    volume = volume_series[0]
    volume_meta = volume.user_dict
    
    # thresholded = volume.threshold(min_holdup, scalars='Density')
    
    # select data inside the column
    column = pv.Cylinder(center=volume.center, direction=(0, 0, 1),
                         height=volume_meta['volume_height'],
                         radius=column_diameter / 2)
    clipped = volume.clip_surface(column, compute_distance=True)

    # select data with holdup > 0.3

    # make opacity a function of holdup
    pl = pv.Plotter()
    pl.open_movie(output_folder / output_name)

    t_text = time_string(framerate, float(volume_meta['frame']))
    print(t_text)
    pl.add_text(t_text, name='time-label')
    pl.add_volume(clipped, cmap=colormap, clim=[0, 1], name='gas-fraction',
                  scalar_bar_args={'title': 'Gas fraction [-]'})
    pl.add_mesh(clipped.outline_corners())
    pl.show(auto_close=False)

    pl.write_frame()

    for new_volume in volume_series[1:]:
        # thresholded = new_volume.threshold(min_holdup, scalars='Density')
        new_meta = new_volume.user_dict
        t_text = time_string(framerate, float(new_meta['frame']))
        print(t_text)
        pl.add_text(t_text, name='time-label')
        clipped = new_volume.clip_surface(column, compute_distance=True)
        pl.add_volume(clipped, cmap=colormap, clim=[0, 1], name='gas-fraction',
                      scalar_bar_args={'title': 'Gas fraction [-]'})
        pl.write_frame()

    pl.close()


def create_3d_planes_animation_pv(
        filenames: Iterable[Path],
        output_folder: Path,
        output_name: str,
        colormap: mplc.Colormap | str = "viridis",
        framerate: int = 22,
        flowrate: float | None = None,
        min_holdup: float = 0.3,
        max_holdup: float = 0.5,
        column_diameter: float = 19.2,
        ):
    colormap = generate_colormap(colormap)

    volume_series = pv.read(filenames)
    volume = volume_series[0]
    volume_meta = volume.user_dict
    
    # thresholded = volume.threshold(min_holdup, scalars='Density')
    
    # select data inside the column
    column = pv.Cylinder(center=volume.center, direction=(0, 0, 1),
                         height=volume_meta['volume_height'],
                         radius=column_diameter / 2)
    clipped = volume.clip_surface(column, compute_distance=True)
    yz_plane = clipped.clip_slab(
        thickness=volume_meta['voxel_size'] * 2,
        normal=(1, 0, 0),
        origin=volume.center)
    xz_plane = clipped.clip_slab(
        thickness=volume_meta['voxel_size'] * 2,
        normal=(0, 1, 0),
        origin=volume.center)
    xy_plane = clipped.clip_slab(
        thickness=volume.meta['voxel_size'] * 2,
        normal=(0, 0, 1),
        origin=volume.center
    )

    # select data with holdup > 0.3

    # make opacity a function of holdup
    pl = pv.Plotter()
    pl.open_movie(output_folder / output_name)

    t_text = time_string(framerate, float(volume_meta['frame']))
    print(t_text)
    pl.add_text(t_text, name='time-label')
    # pl.add_volume(clipped, cmap=colormap, clim=[0, 1], name='gas-fraction',
    #               scalar_bar_args={'title': 'Gas fraction [-]'})
    pl.add_volume(yz_plane, cmap=colormap, clim=[0, 1], name='gas-fraction-yz',
                  scalar_bar_args={'title': 'Gas fraction [-]'})
    pl.add_volume(xz_plane, cmap=colormap, clim=[0, 1], name='gas-fraction-xz',
                  show_scalar_bar=False)
    pl.add_volume(xy_plane, cmap=colormap, clim=[0, 1], name='gas-fraction-xy',
                  show_scalar_bar=False)
    pl.add_mesh(clipped.outline_corners())
    pl.show(auto_close=False)

    pl.write_frame()

    for new_volume in volume_series[1:]:
        # thresholded = new_volume.threshold(min_holdup, scalars='Density')
        new_meta = new_volume.user_dict
        t_text = time_string(framerate, float(new_meta['frame']))
        print(t_text)
        pl.add_text(t_text, name='time-label')
        clipped = new_volume.clip_surface(column, compute_distance=True)
        yz_plane = clipped.clip_slab(
            thickness=volume_meta['voxel_size'] * 2,
            normal=(1, 0, 0),
            origin=volume.center)
        xz_plane = clipped.clip_slab(
            thickness=volume_meta['voxel_size'] * 2,
            normal=(0, 1, 0),
            origin=volume.center)
        xy_plane = clipped.clip_slab(
            thickness=volume.meta['voxel_size'] * 2,
            normal=(0, 0, 1),
            origin=volume.center
        )
        pl.add_volume(yz_plane, cmap=colormap, clim=[0, 1],
                      name='gas-fraction-yz',
                      scalar_bar_args={'title': 'Gas fraction [-]'})
        pl.add_volume(xz_plane, cmap=colormap, clim=[0, 1],
                      name='gas-fraction-xz', show_scalar_bar=False)
        pl.add_volume(xy_plane, cmap=colormap, clim=[0, 1],
                      name='gas-fraction-xy', show_scalar_bar=False)

        pl.write_frame()

    pl.close()


def create_transmission_animation(
        output_folder: Path,
        cameras: list,
        image_series: np.ndarray,
        filename: str | None = None,
        colormap: mplc.Colormap | str | None = None,
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
        colormap (cmap | str | None): Colormap or name of the colormap to use.
            Name should be one of the colormaps in mpl.colormaps. Defaults to
            None, leading to tri_ct_tools.colors.blues9_map being used.
        frame_start (int, optional): Frame number offset for time display. Defaults to 0.
        framerate (int, optional): Framerate of the animation in frames per second.
            Defaults to 22.
        fl (float | None, optional): Flow rate parameter used in filename generation.
            Defaults to None.
        fcounter (int, optional): Frame counter for filename generation. Defaults to 0.

    Returns:
        int: Updated fcounter value (fcounter + 1).
    """
    colormap = generate_colormap(colormap)
    # First dimension of image series is the number of frames
    n_frames = image_series.shape[0]
    fig, axs = plt.subplots(1, len(cameras), figsize=(10, 4), layout="constrained")
    current_images = []
    for i, c in enumerate(cameras):
        current_images.append(axs[i].imshow(image_series[0, i, :, :],
                                            aspect='equal',
                                            cmap=colormap,
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


def plot_quadrants(ax, array, fixed_coord, cmap):
    """For a given 3d *array* plot a plane with *fixed_coord*, using four quadrants."""
    nx, ny, nz = array.shape
    index = {
        'x': (nx // 2, slice(None), slice(None)),
        'y': (slice(None), ny // 2, slice(None)),
        'z': (slice(None), slice(None), nz // 2),
    }[fixed_coord]
    plane_data = array[index]

    n0, n1 = plane_data.shape
    quadrants = [
        plane_data[:n0 // 2, :n1 // 2],
        plane_data[:n0 // 2, n1 // 2:],
        plane_data[n0 // 2:, :n1 // 2],
        plane_data[n0 // 2:, n1 // 2:]
    ]

    min_val = array.min()
    max_val = array.max()

    cmap = plt.get_cmap(cmap)

    for i, quadrant in enumerate(quadrants):
        facecolors = cmap((quadrant - min_val) / (max_val - min_val))
        if fixed_coord == 'x':
            Y, Z = np.mgrid[0:ny // 2, 0:nz // 2]
            X = nx // 2 * np.ones_like(Y)
            Y_offset = (i // 2) * ny // 2
            Z_offset = (i % 2) * nz // 2
            ax.plot_surface(X, Y + Y_offset, Z + Z_offset, rstride=1, cstride=1,
                            facecolors=facecolors, shade=False)
        elif fixed_coord == 'y':
            X, Z = np.mgrid[0:nx // 2, 0:nz // 2]
            Y = ny // 2 * np.ones_like(X)
            X_offset = (i // 2) * nx // 2
            Z_offset = (i % 2) * nz // 2
            ax.plot_surface(X + X_offset, Y, Z + Z_offset, rstride=1, cstride=1,
                            facecolors=facecolors, shade=False)
        elif fixed_coord == 'z':
            X, Y = np.mgrid[0:nx // 2, 0:ny // 2]
            Z = nz // 2 * np.ones_like(X)
            X_offset = (i // 2) * nx // 2
            Y_offset = (i % 2) * ny // 2
            ax.plot_surface(X + X_offset, Y + Y_offset, Z, rstride=1, cstride=1,
                            facecolors=facecolors, shade=False)


def figure_3D_array_slices(array, cmap=None):
    """Plot a 3d array using three intersecting centered planes."""
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_box_aspect(array.shape)
    plot_quadrants(ax, array, 'x', cmap=cmap)
    plot_quadrants(ax, array, 'y', cmap=cmap)
    # plot_quadrants(ax, array, 'z', cmap=cmap)
    return fig, ax


def create_3d_planes_mpl(filenames):
    volume_data, metadata = read_vtk(filenames[0])
    volume_data = volume_data[:, :, 10:-10]     # cut off top and bottom 10 px
    n_colors = 256
    viridis_big = mpl.colormaps['viridis'](np.linspace(0, 1, n_colors))
    viridis_rgb = viridis_big[..., :3]
    viridis_a = np.expand_dims(np.linspace(0, 1, n_colors), -1)
    viridis_rgba = np.concatenate([
        viridis_rgb,
        viridis_a
    ], axis=-1)

    viridis_new = ListedColormap(viridis_rgba)
    
    figure_3D_array_slices(volume_data, cmap=viridis_new)
    plt.show()


if __name__ == "__main__":
    source_folder = Path(R"U:\Xray RPT ChemE\X-ray\Xray_data\2025-06-26 Rik"
                         R"\25_reconstruction_sc_bhc\1500x1500Crop_30lmin_150kV_22Hz")
    filenames = [
        # source_folder / f"recon_loss_time-resolved_frame-{i}.vtk" for i in range(51, 271)
        source_folder / "recon_loss_time-resolved_frame-55.vtk"
    ]
    output_folder = Path(R"D:\XRay\Animations")
    output_name = "2025-06-26_30lmin_150kV_22Hz.mp4"
    create_3d_planes_mpl(filenames)
