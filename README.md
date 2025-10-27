# 3ct-tools
Python tools for analysis and processing of data from the 3-angle X-ray tomography setup in the chemical engineering department at Delft University of Technology

# Setup
## Requirements / environment
... To be filled ...

## Building and including the package
Building the project relies on setuptools and `build`. If you're not sure if 
`build` is installed, run
```bash
python -m pip install --upgrade build
```
Then, build the project with
```bash
python -m build
```

And to have the project recognized as a package locally, use
```bash
pip install --editable .
```

It is unfortunate to use `pip` inside of a mainly conda-managed environment, but
for this application there is no straightforward conda alternative. While `conda
develop` exists, this just adds a folder to PATH, not quite the same as a 
functional install.

Later on the project might be posted on PyPI and maybe even conda, but for now
that's very far away.
