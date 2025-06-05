import torch
import gpytorch
from tqdm.notebook import tqdm
import CustomKernels.LaplacianKernel as LK

import CustomKernels.LaplacianKernelUncertainty as LKUncertain
import CustomKernels.UncertainKernel as UncertainKernel
import CustomKernels.UncertainMeanConstant as UncertainMean

import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_dims, uncertain):
        super().__init__(train_x, train_y, likelihood)
        if uncertain:
            self.mean_module = UncertainMean.ConstantMean()
            self.covar_module = (gpytorch.kernels.ScaleKernel(LKUncertain.LaplacianKernel()) + gpytorch.kernels.ScaleKernel(UncertainKernel.UncertainKernel(ard_num_dims=num_dims))).cuda()
        else:
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = (gpytorch.kernels.ScaleKernel(LK.LaplacianKernel()) + gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=num_dims))).cuda()

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def load_model(EDFAID, num_dims, uncertain):

    likelihoods = []
    models = []

    path = "CascadedPowerModelling/SavedModels/"
    if uncertain:
        path = path+"NIGP/EDFA"
    else:
        path = path+"GP/EDFA"

    train_data = torch.load(path+EDFAID+"_trainingData.pt")
    state_dict = torch.load(path+EDFAID+".pth")

    for i in range(16):
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = ExactGPModel(train_data[i][0], train_data[i][1], likelihood, num_dims, uncertain)
        likelihoods.append(likelihood)
        models.append(model)

    model = gpytorch.models.IndependentModelList(*models).to(device)
    likelihood = gpytorch.likelihoods.LikelihoodList(*[model.likelihood for model in models]).to(device)

    model.load_state_dict(state_dict)

    return model, likelihood



def batchPreds(model, likelihood, testX, numTrain):

    splitTesting = torch.split(testX, numTrain)
    allPreds = [[] for x in range(16)]
    allvars = [[] for x in range(16)]
    
    for batch in splitTesting:

        if len(batch) != numTrain:
            break
        with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.lazily_evaluate_kernels(False):
            observed_pred = likelihood(*model(*[batch for x in range(16)]))

        for i, submodel, prediction in zip(range(16), model.models, observed_pred):
            preds = prediction.mean.cpu().tolist()
            variance = prediction.variance.cpu().tolist()
            allPreds[i].append(preds)
            allvars[i].append(variance)

    npPreds = np.array(allPreds)
    npVars = np.array(allvars)

    flattenedPreds = [arr.flatten() for arr in npPreds]
    flattenedVars = [arr.flatten() for arr in npVars]

    return np.array(flattenedPreds).T, np.array(flattenedVars).T



def make_predictions(model, likelihood, data, uncertain, EDFAID=None):

    model.eval()
    likelihood.eval()

    if uncertain:
        path = "CascadedPowerModelling/SavedModels/"
        train_data = torch.load(path+"NIGP/EDFA"+EDFAID+"_trainingData.pt")

        predsFormatted, variancesFormatted = batchPreds(model, likelihood, data, len(train_data[0][0]))

        return predsFormatted, variancesFormatted

    allPreds = []
    variances = []

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood(*model(*[data for x in range(16)]))


    for submodel, prediction in zip(model.models, observed_pred):

        preds = prediction.mean.cpu().tolist()
        allPreds.append(preds)
        variances.append(prediction.variance.cpu().numpy())

    predsFormatted = np.array(allPreds).T
    variancesFormatted = np.array(variances).T

    return predsFormatted, variancesFormatted