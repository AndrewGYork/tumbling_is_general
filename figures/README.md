Written in python 3.9.9
Dependency versions:
Scipy 1.7.2
Numpy 1.21.4
Matplotlib 3.7.1
Pandas 1.5.3

Hopefully platform-independent, but only tested on Mac

The scripts in the figures folder import .csv files generated either by simulation code in the figures folder or by pre-processing scripts in the source_data folder. Plotting scripts are separate from simulation scripts. Running the scripts in numerical order within each folder will result in generation of all necessary files except for experimental data.

Note that the raw experimental data to generate these figures is located at: 
https://doi.org/10.5281/zenodo.10028240

To get the relative paths to work with the experimental data, you must first download the data from zenodo, unzip it, and place the "source_data" folder in a "tumbling_temp" directory in the same folder as the repository folder. The "tumbling_temp" folder is also used to keep the intermediate images for generating GIFs separate from the version controlled components, and it will be created by running the figure generation code for the simulations.