import torch
import gpytorch
from gpytorch.mlls import SumMarginalLogLikelihood
from sklearn.preprocessing import MinMaxScaler

import numpy as np
import pandas as pd
from tqdm import tqdm


import kernels.LaplacianKernel as LK
import kernels.LaplacianKernelUncertainty as LKUncertain
import kernels.UncertainKernel as UncertainKernel
import kernels.UncertainMeanConstant as UncertainMean
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
    path = "models/"
    if uncertain:
        path = path+"nigp/EDFA"
    else:
        path = path+"gp/EDFA"

    train_data = torch.load(path+EDFAID+"_trainingData.pt")
    state_dict = torch.load(path+EDFAID+".pth")

    for i in range(len(train_data)):
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = ExactGPModel(train_data[i][0], train_data[i][1], likelihood, num_dims, uncertain)
        likelihoods.append(likelihood)
        models.append(model)

    model = gpytorch.models.IndependentModelList(*models).to(device)
    likelihood = gpytorch.likelihoods.LikelihoodList(*[model.likelihood for model in models]).to(device)

    model.load_state_dict(state_dict)

    return model, likelihood, train_data

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

def make_predictions(inputs, variances, uncertain, EDFAID, device):

    model, likelihood, train_data = load_model(EDFAID, inputs.shape[-1], uncertain)

    inputs_tensor = torch.tensor(inputs, device=device)

    model.eval()
    likelihood.eval()

    if uncertain:
        variance_tensor = torch.tensor(variances, device=device)
        inputsZipped = torch.stack((inputs_tensor, variance_tensor), dim=1)

        if len(inputsZipped) < len(train_data[0][0]):

            numToPad = len(train_data[0][0]) - len(inputsZipped)
            dimensions = len(inputsZipped[0][0])

            padding = torch.tensor([[[0]*dimensions]*2]*numToPad, device=inputsZipped.device, dtype=inputsZipped.dtype)
            paddedData = torch.concat([inputsZipped, padding])

            predsFormatted, variancesFormatted = batchPreds(model, likelihood, paddedData, len(train_data[0][0]))

            return predsFormatted[:len(inputsZipped)], variancesFormatted[:len(inputsZipped)]

        predsFormatted, variancesFormatted = batchPreds(model, likelihood, inputsZipped, len(train_data[0][0]))
        return predsFormatted, variancesFormatted

    allPreds = []
    variances = []

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood(*model(*[inputs_tensor for x in range(16)]))

    for submodel, prediction in zip(model.models, observed_pred):
        preds = prediction.mean.cpu().tolist()
        allPreds.append(preds)
        variances.append(prediction.variance.cpu().numpy())

    predsFormatted = np.array(allPreds).T
    variancesFormatted = np.array(variances).T

    return predsFormatted, variancesFormatted





def preprocess_data(inputPowers, outputPowers, settings, onChannels, uncertain):

    outputPowers = outputPowers * onChannels
    inputPowers = inputPowers * onChannels

    inputPowers[inputPowers == 0] = inputPowers.min().min() - 5
    outputPowers[outputPowers == 0] = outputPowers.min().min() - 5

    combinedInputs = pd.concat([inputPowers, settings], axis=1)

    inputPowersArr = combinedInputs.to_numpy()
    outputPowersArr = outputPowers.to_numpy()
    
    scalerIn = MinMaxScaler()
    scaledInputs = scalerIn.fit_transform(inputPowersArr)


    if uncertain:
        initialVariance = np.ones_like(scaledInputs) * 0
    else:
        initialVariance = None

    return scaledInputs, outputPowersArr, initialVariance



def split_data(inputs, outputs, numTraining, testIndex, uncertain=False, variances=None):

    if uncertain and variances is None:
        raise ValueError("Variances must be provided if uncertain is True")

    X_train, y_train = inputs[:numTraining], outputs[:numTraining]
    X_test, y_test = inputs[testIndex:], outputs[testIndex:]

    X_train, y_train = torch.tensor(X_train).to(device), torch.tensor(y_train).to(device)
    X_test, y_test = torch.tensor(X_test).to(device), torch.tensor(y_test).to(device)

    if not uncertain:
        return X_train, y_train, X_test, y_test

    train_std, test_std = variances[:numTraining], variances[testIndex:]
    train_std, test_std = torch.tensor(train_std).to(device), torch.tensor(test_std).to(device)

    inputsZipped = torch.stack((X_train, (train_std)), dim=1)
    testingZipped = torch.stack((X_test, (test_std)), dim=1)

    return inputsZipped, y_train, testingZipped, y_test



def train_model(X_train, y_train, device, uncertain):


    likelihoods = []
    models = []
    outputDimensions = y_train.shape[1]
    inputDimensions = X_train.shape[-1]

    train_data = []
    for x in range(outputDimensions):
        train_data.append((X_train, y_train[:,x]))

    for i in range(outputDimensions):
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = ExactGPModel(train_data[i][0], train_data[i][1], likelihood, inputDimensions, uncertain)
        likelihoods.append(likelihood)
        models.append(model)

    model = gpytorch.models.IndependentModelList(*models).to(device)
    likelihood = gpytorch.likelihoods.LikelihoodList(*[model.likelihood for model in models]).to(device)

    mll = SumMarginalLogLikelihood(likelihood, model)


    training_iterations = 1000

    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    for i in tqdm(range(training_iterations),ncols=200):
        optimizer.zero_grad()
        with gpytorch.settings.lazily_evaluate_kernels(False):
            output = model(*model.train_inputs)
            loss = -mll(output, model.train_targets)
            loss.backward()

        optimizer.step()


    return model, likelihood, train_data



def test_model(model, likelihood, X_test, y_test, numTraining, uncertain):

    model.eval()
    likelihood.eval()

    allPreds = []
    lowers = []
    uppers = []
    variances = []

    if uncertain:
        predsFormatted, variancesFormatted = batchPreds(model, likelihood, X_test, numTraining)
    else:
        with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.lazily_evaluate_kernels(False):
            observed_pred = likelihood(*model(*[X_test for x in range(16)]))

        for submodel, prediction in zip(model.models, observed_pred):

            preds = prediction.mean.cpu().tolist()
            lower, upper = prediction.confidence_region()
            allPreds.append(preds)
            variances.append(prediction.variance.cpu().numpy())

        predsFormatted = np.array(allPreds).T
        variancesFormatted = np.array(variances).T

    actual = y_test.cpu().numpy()[:len(predsFormatted)]

    l1 = np.mean(np.abs((actual - predsFormatted)))
    l2 = np.mean(np.square((actual- predsFormatted)))

    return l1, l2, predsFormatted, variancesFormatted


def save_model(model, training_data, uncertain, EDFAID):

    if uncertain:
        path = "models/nigp/EDFA"
    else:
        path = "models/gp/EDFA"

    torch.save(model.state_dict(), path+EDFAID+".pth")
    torch.save(training_data, path+EDFAID+"_trainingData.pt")