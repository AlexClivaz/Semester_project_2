# How to prepare data

Only support Python 3.

# Step 0: Download TUH data
Please download v1.5.2 from From https://www.isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml into [Path to Raw data]

## Step 1: Build Data
build_data.py extracts all the .edf files from all folders and extracts them according to their seizure type into pickle format (types of interest can be specified with seizure_types).