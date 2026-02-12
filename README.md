# 3ct-tools
Python tools for analysis and processing of data from the 3-angle X-ray 
tomography setup in the chemical engineering department at Delft University of 
Technology.

# Setup
It is assumed that a user of this repository is familiar with the following:
- conda environment management
- git version control
- GitHub collaboration

If you are unfamiliar with any of these topics, please spend a bit of time 
following tutorials and watching YT / TikTok videos about how they work. I can 
also recommend looking up the CodeRefinery guides on the topics, they are great! 
You could also try to vibecode your way through the endeavour, but nothing beats 
actually knowing what you're doing, at least a little bit.

## Requirements / environment
In order to use the files here, you will usually need to create the relevant 
conda environment with all packages available. The project requirements can be 
found in `environment.yaml`. The easiest way to create an environment out of 
this is through
```bash
conda env create -f environment.yaml
```
which will install everything you need in the environment tri-ct-tools.

This environment can then be activated
```bash
conda activate tri-ct-tools
```

## Dealing with notebooks
The main workhorse for quick image analysis is the Jupyter notebook. 
Unfortunately, Jupyter notebooks and git don't play nice together. 

If you're going to run any of the notebooks and make git commits, make sure to run the 
following command from the folder root: 

```bash
pre-commit install
```
You only have to run it once, then git knows what to do. If you ever delete the 
`.git` folder you'll have to run it again. But in that case you likely have an 
idea what you're doing.

This registers the 'pre-hook' defined in `.pre-commit-config.yaml`, essentially 
running `nbstripout` before each commit. `nbstripout` strips out all notebook 
outputs, making it much easier for multiple people to work on the notebooks.

Note: This is a thing that deletes outputs from notebooks when they are 
committed (or technically right before that. You know, pre-commit). Sometimes, 
this means that when you press / enter a commit command, it fails because this 
guy needs to do its thing. No big deal, just stage the file again, commit again 
and it will all be fine.

**Do not commit notebooks without having ran this! That would make everyones** 
**lives miserable.**

## Building and including the package
To easily use the package inside of the notebooks, you will need to build and 
install it. This requires some setup and a few commands, explained below.

Building the project relies on setuptools and `build`. If you're not sure if 
`build` is installed, run
```bash
python -m pip install --upgrade build
```
Then, build the project with
```bash
python -m build
```

And to have the project recognized as a package locally, use the following 
from the folder root while the `tri-ct-tools` conda environment is active:
```bash
pip install --editable .
```

It is unfortunate to use `pip` inside of a `conda`-managed environment, but
for this application there is no straightforward conda alternative. While 
`conda develop` exists, this just adds a folder to `PATH`, not quite the same as a 
functional install.

The geometry-based functionalities in the package rely on the _CaTE_ package, 
which should also be included in the environment. Easiest way to do so is through
```shell
cd path/to/tri_ct_tools
cd ../
git clone https://github.com/RikVolger/CaTE/tree/develop
cd cate
conda activate tri-ct-tools
conda develop .
```

Later on the project might be posted on PyPI and maybe even conda which would 
make installation a breeze, but for now that's very far away.

If you notice that the current implemented features are not enough for your 
needs, get in touch and we'll see if we can improve some things. If you have 
ideas for improvements, don't hesitate to open an issue or pull request. 

If you had no idea what that meant, look up some more info on GitHub 
collaboration.

## Using the tools
### Image inspection during measurement
For this purpose, the notebooks `check_tiffs.ipynb` and 
`check_rolling_mean.ipynb` are useful. More can be added for specific goals.
`check_tiffs` can be used to quickly show some images, horizontal / vertical 
lines through them or relative images.

### Image pre- and post-processing
After a measurement series, the images obtained are not ready for proper 
analysis yet. There are dead pixels in the raw images, there are dark offsets 
that need to be corrected, there is still cross-scatter in the images and there
might be beam hardening artefacts. All of these can be corrected with the 
appropriate scripts in this package.

Generally speaking, the files will rely on a `yaml` file to indicate what they 
should do to what files and where to put the results. The scripts themselves 
should not change much. There are templates for the `yaml` files, the best way 
to use them is by copying them close to your actual data, and changing them to 
reflect your setup. Then point the appropriate scripts to where your `yaml` 
files now live.

#### Dead pixel corrections
The script `preprocess.dead_pixels` is used to correct the dead pixel lines that 
the detectors produce. It uses two inputs: `preprocess/dead_pixel_lines.yaml` and 
`inputs/dead_pixel_multisource.yaml` (or ideally, a file following that template 
living elsewhere). 

The first file (`preprocess/dead_pixel_lines.yaml`) should be static. The 
distance between the dead pixel lines is (so far) always the same. If new lines 
or single pixels appear, they should be added to this.

The second file (`inputs/dead_pixel_multisource.yaml`) is the file that you use 
to tell the script where to look for the raw images, where to copy them to and 
what the offset of the dead pixel pattern is. This pattern sometimes shifts one 
or two pixels, resulting in failure of the dead pixel correction procedure. If 
this is the case (check by inspecting the images however you prefer), the 
`offsets` in this file need to be changed.

The first inputs in this `yaml` are `input_folders` and `output_folders`. No 
checks are performed in terms of names, all subfolders found in each input folder 
will be created in the corresponding output folder and filled with dead pixel 
corrected data. That also means that these lists should have the same length, 
otherwise weird things will happen.

Then, the file allows (and would encourage, if it could) providing multiple dark 
measurement folders. For each measurement found in the input folders, the 'best 
fitting' dark measurement will be found. Best fitting is determined as the dark 
measurement at the same framerate (essential), and with the same VROI setting 
(ideally) with the least time between dark and real measurement. If a same VROI 
is not found, the relevant section is cut from a Mag0 dark measurement, if 
available. If a dark measurement at the same framerate is unavailable, the 
script will raise an error.

There is the flag `copy_raw`, used to indicate if next to each output folder, a 
new `01_raw` folder should be created. When copying data directly from the 
measurement pc to the project drive, this should be true. When (p)reprocessing 
data that is already on the project drive, you likely want it to be false.

The `avg_only` flag allows you to save only the averaged preprocessed image, 
saving a lot of data storage. Use this if you are only interested in the 
time-averaged measurements.

The `subtract_dark` is a currently unused flag that is always true. It was 
intended for cases where you don't wish to correct for the dark signal offset 
but it was then decided that that would be madness so it was left unimplemented.

The `mirror` flag allows you to swap the left-right direction of the images. 
Beware that this also mirrors the setup for subsequent tomography steps.

#### Cross-scatter corrections
The script `preprocess.scatter` allows for the correction of cross-scatter 
through independent cross-scatter measurements. It relies on a `scatter.yaml` 
file for the inputs. The script itself should generally remain as-is, unless 
you're implementing new features.

The first input is the most important, the list of `roots` - the base folders. 
The script will attempt to scatter correct all subdirectories in there that do 
not have the word 'scatter' in their name. Don't worry if you have folders in 
there that don't have corresponding scatter measurements (e.g. Alignment), they 
will simply be skipped.

The `cameras` and `scatter_IDs` need to be paired - The first scatter ID should 
match the first camera number. The scatter IDs will be used to try and find the 
scatter measurements corresponding to a certain measurement, currently following 
a very rigid pattern. Will still be improved in the near future.

When the flag `average` is true, the scatter-correction will be applied to the 
averaged images. When it is false, the script has no idea what to do and will 
likely crash.

`frames` indicates the frameranges for the different types of measurements. And 
finaly `img` is used to indicate the image size expected. This means that if you 
measure at different VROI settings, you need to make different config files.

#### Geometry optimization
A prerequisite for this step is to follow the needle calibration procedure from 
https://github.com/RikVolger/fluidized-bed-ct to have a base geometry to 
optimize.

In this case, the regular scheme of a `yaml` inputing settings and a script 
doing the things does not hold. Here, the settings are put in the file itself.
All the way at the bottom, four paths are required. The first is the path to 
the geometry as it came from the needle calibration. The second is the root path 
for the next two paths, the path to a measurement of a column full of water and 
a measurement of an empty column.

The script will optimize the geometry with some tiny shifts, report on this 
optimization and save the new geometry as `bhc_optimized_geom.npy`. This is the 
file that you should use as the geometry in the next steps.

#### Beam hardening correction
This once again does follow the old `yaml` setup for input settings. The file 
used as template for future setups is `beam_hardening_corrections.yaml`. 

The file starts with `geom_path`, pointing to the geometry generated from 
geometry optimization. The `det` entry provides some detector specs. If VROI is
used, this should be reflected in the `rows` field.

The frameranges indicate what ranges to load if `average.tif` files are not 
found. The `ROI` setting is an important one here, indicating the start and stop 
line of the region to use for the beam hardening correction. If there are probes 
or tubes in the way, this can affect the precision.

`cameras` indicates which cameras to appy the correction to. Note that here, we 
use 0-based indexing (so all cameras is [0, 1, 2], only camera 2 is \[1\])

What follows is a list of series - the measurements to correct and the 
reference measurements to use for that. Each entry gets appointed a name for 
easier referencing, and will point to a full and empty measurement to use. The 
field empty_copy indicates where a copy of the empty measurement should be 
placed, as the beam hardening approach is not performed on that.
`meas` contains a list of all measurements with the intended output folder.