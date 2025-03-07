Code to reproduce results in : ...



Usage:

Each individual EDFA model is found in the notebooks. To evaluate cascaded results, run EDFA6_model.ipynb with the uncertain flag to True for NIGP and False for standard GP modelling.

All notebooks can be run sequentially from 1 to 6 with the Orchestration.py file.



Folders:

CustomKernels - custom Gpytorch kernels for allowing uncertainty as an input into the model

data - input, output, config, and EDFA settings for each EDFA and fibre in the cascade

Functions - helper functions to load pre-trained models and train new ones

SavedModels - saved config for trained models used to cascade through previous models
