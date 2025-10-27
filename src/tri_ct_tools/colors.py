from matplotlib.colors import ListedColormap
from cycler import cycler


blues9 = [
    '#ef3b2c',
    '#f7fbff',
    '#deebf7',
    '#c6dbef',
    '#9ecae1',
    '#6baed6',
    '#4292c6',
    '#2171b5',
    '#08519c',
    '#08306b',
]

blues9_map = ListedColormap(blues9[::-1], name='blues9')

tud_cycler = cycler(color=[
    "#00a6d6",
    "#c3312f",
    "#00a390",
    "#f6d37d",
    "#eb7246",
    "#017188"
])
