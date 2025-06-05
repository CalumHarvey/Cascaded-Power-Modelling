Code to reproduce results in : ...



Usage:

The code to train each individual EDFA model can be found in GP_functions for GP and NIGP, and ANN_Functions for the ANN approaches.

To run model training, execute functions located in either ANN_modelTraining.py or GP_modelTraining.py where the uncertain flag is used to differentiate between GP and NIGP cascade.


Folders:

CustomKernels - custom Gpytorch kernels for allowing uncertainty as an input into the model

data - input, output, config, and EDFA settings for each EDFA and fibre in the cascade

SavedModels - saved config for trained models used to cascade through previous models
