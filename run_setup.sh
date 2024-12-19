#!/bin/bash -l
eval "$(conda shell.bash hook)"
test=$(conda info --envs | grep 'phys_effects' | cut -d' ' -f1)

export CONDA_ALWAYS_YES="true"

echo 'Creating a new conda environment called phys_effects'
# Create a new conda environment
conda create -n phys_effects python=3.8

# Install some important packages
conda install -c conda-forge cftime
conda install -c conda-forge -n phys_effects iris
conda install -c conda-forge -n phys_effects mo_pack
conda install -c conda-forge -n phys_effects h5py
conda install -n phys_effects sphinx # For documentation
conda install -c conda-forge -n phys_effects gdal
conda install -c conda-forge -n phys_effects geopandas
conda install -c conda-forge -n phys_effects pyogrio
conda install -c conda-forge -n phys_effects fiona=1.8.19
conda install -c conda-forge -n phys_effects rasterstats
conda install -c conda-forge -n phys_effects mapclassify
conda install -c conda-forge -n phys_effects regionmask cartopy pygeos
conda install -c conda-forge -n phys_effects seaborn
conda install -c conda-forge -n phys_effects ascend
conda install -c conda-forge -n phys_effects cdo
conda install -c conda-forge -n phys_effects tabulate
conda install -c conda-forge -n phys_effects pyproj
conda install -c conda-forge -n phys_effects adjustText

conda activate phys_effects

#jupyter labextension install @jupyterlab/geojson-extension

unset CONDA_ALWAYS_YES
