# pypluto scripts

This repository contains python scripts to plot useful quantities of hydrodinamics simulations data produced with the PLUTO software by using the pypluto library.
These scripts are not refined for production, but rather intended for personal use.

## Contents

- en.py                : Computes actual energy and mass of PLUTO simulation initial condition based on tracers. Computes also maximum and minimum pressure and density in both code an physical units.
- fill_factor.ipynb    : Computes a filling factor based on the number of cells composed by a percentage of a tracer with respect to a predefined solid volume. The script requires a "r.bin" file containing a 3D meshgrid consisting of radial distance values from a point.
- r_data.py            : Produce the "r.bin" files, based on the grid of a pluto 3D simulation file.
- panel_plot.ipynb     : Plots sections of density and temperature from different PLUTO 3D cartesian coordinates simulation files.
- plot2dSph.py         : Plots different quantities of 2D spherical coordinates pluto simulation files and makes animated plots.
- plt_cut.py           : Plots different quantities of 3D cartesian coordinates pluto simulation files and makes animated plots.
- save_slice.py        : Saves binary files containing slices of some quantities of PLUTO simulation files.

The original datasets used in these scripts are not included, due to size or availability constraints.

## Status

This is a personal, static collection of scripts and is not intended for active development or external contributions.
