from torch import nn
import torch
import pandas as pd
import numpy as np
import gpytorch
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import CustomKernels.LaplacianKernel as LK

import CustomKernels.LaplacianKernelUncertainty as LKUncertain
import CustomKernels.UncertainKernel as UncertainKernel
import CustomKernels.UncertainMeanConstant as UncertainMean

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def trainModel(inputPowers, outputPowers, settings, perms, uncertain, numTraining=300):

    onChannels = perms.copy()

    onChannels[onChannels > -1] = 1
    onChannels[onChannels == -1] = 0

    outputPowers = outputPowers * onChannels
    inputPowers = inputPowers * onChannels

    inputPowers[inputPowers == 0] = inputPowers.min().min() - 5
    outputPowers[outputPowers == 0] = outputPowers.min().min() - 5

    combinedInputs = pd.concat([inputPowers, settings], axis=1)


    inputPowersArr = combinedInputs.to_numpy()
    outputPowersArr = outputPowers.to_numpy()
    
    scalerIn = MinMaxScaler()
    scaledInputs = scalerIn.fit_transform(inputPowersArr)

    testIndex = 300

    X_train, y_train = scaledInputs[:numTraining], outputPowersArr[:numTraining]
    X_test, y_test = scaledInputs[testIndex:600], outputPowersArr[testIndex:600]

    X_train, y_train = torch.tensor(X_train).to(device), torch.tensor(y_train).to(device)
    X_test, y_test = torch.tensor(X_test).to(device), torch.tensor(y_test).to(device)

    train_std, test_std = np.ones_like(X_train.cpu().numpy()) * 0, np.ones_like(X_test.cpu().numpy()) * 0
    train_std, test_std = torch.tensor(train_std).to(device), torch.tensor(test_std).to(device)

    inputsZipped = torch.stack((X_train, (train_std)), dim=1)
    TestingZipped = torch.stack((X_test, (test_std)), dim=1)


    class ExactGPModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super().__init__(train_x, train_y, likelihood)
            if uncertain:
                self.mean_module = UncertainMean.ConstantMean()
                self.covar_module = (gpytorch.kernels.ScaleKernel(LKUncertain.LaplacianKernel()) + gpytorch.kernels.ScaleKernel(UncertainKernel.UncertainKernel(ard_num_dims=17))).cuda()
            else:
                self.mean_module = gpytorch.means.ConstantMean()
                self.covar_module = (gpytorch.kernels.ScaleKernel(LK.LaplacianKernel()) + gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=17))).cuda()

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


    likelihoods = []
    models = []
    if uncertain:
        train_data = [(inputsZipped, y_train[:,0]), (inputsZipped, y_train[:,1]), (inputsZipped, y_train[:,2]), (inputsZipped, y_train[:,3]), (inputsZipped, y_train[:,4]), (inputsZipped, y_train[:,5]), (inputsZipped, y_train[:,6]), (inputsZipped, y_train[:,7]), (inputsZipped, y_train[:,8]), (inputsZipped, y_train[:,9]), (inputsZipped, y_train[:,10]), (inputsZipped, y_train[:,11]), (inputsZipped, y_train[:,12]), (inputsZipped, y_train[:,13]), (inputsZipped, y_train[:,14]), (inputsZipped, y_train[:,15])]
    else:
        train_data = [(X_train, y_train[:,0]), (X_train, y_train[:,1]), (X_train, y_train[:,2]), (X_train, y_train[:,3]), (X_train, y_train[:,4]), (X_train, y_train[:,5]), (X_train, y_train[:,6]), (X_train, y_train[:,7]), (X_train, y_train[:,8]), (X_train, y_train[:,9]), (X_train, y_train[:,10]), (X_train, y_train[:,11]), (X_train, y_train[:,12]), (X_train, y_train[:,13]), (X_train, y_train[:,14]), (X_train, y_train[:,15])]


    for i in range(16):
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = ExactGPModel(train_data[i][0], train_data[i][1], likelihood)
        likelihoods.append(likelihood)
        models.append(model)

    model = gpytorch.models.IndependentModelList(*models).to(device)
    likelihood = gpytorch.likelihoods.LikelihoodList(*[model.likelihood for model in models]).to(device)


    from gpytorch.mlls import SumMarginalLogLikelihood

    mll = SumMarginalLogLikelihood(likelihood, model)


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


    training_iterations = 1000
    l1loss_fn = nn.L1Loss(reduction='mean')

    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

    for i in tqdm(range(training_iterations),ncols=200):
        optimizer.zero_grad()
        with gpytorch.settings.lazily_evaluate_kernels(False):
            output = model(*model.train_inputs)
            loss = -mll(output, model.train_targets)
            loss.backward()

        optimizer.step()


    model.eval()
    likelihood.eval()

    l1loss_fn = nn.L1Loss(reduction='mean')

    allPreds = []
    lowers = []
    uppers = []
    variances = []

    if uncertain:
        predsFormatted, variancesFormatted = batchPreds(model, likelihood, TestingZipped, numTraining)
    else:
        with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.lazily_evaluate_kernels(False):
            observed_pred = likelihood(*model(*[X_test for x in range(16)]))


        for submodel, prediction in zip(model.models, observed_pred):

            preds = prediction.mean.cpu().tolist()
            lower, upper = prediction.confidence_region()
            allPreds.append(preds)
            lowers.append(lower.cpu().numpy())
            uppers.append(upper.cpu().numpy())
            variances.append(prediction.variance.cpu().numpy())

        predsFormatted = np.array(allPreds).T
        lowersFormatted = np.array(lowers).T
        uppersFormatted = np.array(uppers).T
        variancesFormatted = np.array(variances).T

    activeChannels = perms + 1
    activeChannels[activeChannels > 0 ] = 1
    activeChannels[activeChannels == 0] = None

    error = np.mean(np.abs((y_test.cpu().numpy()[:len(predsFormatted)] - predsFormatted)))

    path = "CascadedPowerModelling/SavedModels/"
    if uncertain:
        torch.save(model.state_dict(), path+"NIGP/EDFA1.pth")
        torch.save(train_data, path+"NIGP/EDFA1_trainingData.pt")
    else:
        torch.save(model.state_dict(), path+"GP/EDFA1.pth")
        torch.save(train_data, path+"GP/EDFA1_trainingData.pt")
    
    return error