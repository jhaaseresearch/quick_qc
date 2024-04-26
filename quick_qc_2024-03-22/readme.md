
Quick QC Tool
This tool is designed to perform quick quality control (QC) on atmospheric profiles and generate visual and text-based summaries of the QC results. It includes functionality to load profiles from netCDF files, perform QC checks, and create static and interactive plots of the profiles.

Example Usage
python quick_qc.py config.yaml

Algorithm Description
Read in target parameters and settings from the configuration file.
Load desired profiles to perform quick QC on into a Pandas DataFrame.
Perform quick QC on profiles.
Generate text-based results from QC.
Make a summary refractivity plot of all profiles.
Make an interactive version of the above plot.

Instructions
Identify flights and aircraft that you want to perform quick QC on and modify the configuration file accordingly.
Confirm the contents of the 'nret' folder for those profiles are reasonable.
Run the quick_qc.py script.

Outputs
A text file containing a list of suspect profiles and some summary statistics.
A static refractivity plot of all QCed profiles.
An interactive refractivity plot, which can be opened in a web browser.

Dependencies
This tool is optimized to run in the (aro_env) conda environment on 'claron' for best results.
Uses 3rd party Python packages: matplotlib, numpy, pandas, yaml, plotly.

Source
Source path: /ags/projects/hiaper/code_qc/quick_qc/quick_qc_2024-03-22/py/quick_qc.py
Author: Noah Barton (nbarton@ucsd.edu)
Last Modified: 2024-03-14 (nbarton, moved source path)
Created: 2024-02-05 (nbarton)