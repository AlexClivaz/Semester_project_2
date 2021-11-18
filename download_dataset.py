import kaggle

kaggle.api.authenticate()
kaggle.api.dataset_download_files('psyryuvok/the-tuh-eeg-seizure-corpus-tusz-v152', unzip=True)
