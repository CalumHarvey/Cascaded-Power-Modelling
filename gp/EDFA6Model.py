from gp import EDFA_ModelLoad, Fibre_ModelLoad

import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from sklearn.preprocessing import MinMaxScaler

import gpytorch
import kernels.LaplacianKernel as LK
import kernels.LaplacianKernelUncertainty as LKUncertain
import kernels.UncertainKernel as UncertainKernel
import kernels.UncertainMeanConstant as UncertainMean

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def trainModel(inputPowers, outputPowers, settings, perms, uncertain, numTraining=300):

    # Preprocessing data
    onChannels = perms.copy()

    onChannels[onChannels > -1] = 1
    onChannels[onChannels == -1] = 0

    outputPowers = outputPowers * onChannels
    inputPowers = inputPowers * onChannels

    inputPowers[inputPowers == 0] = inputPowers.min().min() - 5
    outputPowers[outputPowers == 0] = outputPowers.min().min() - 5
    
    EDFA11_setting = settings["0"]

    combinedInputs = pd.concat([inputPowers, EDFA11_setting], axis=1)

    # EDFA 1 MODEL 
    model, likelihood = EDFA_ModelLoad.load_model("1", 17, uncertain)

    scalerIn = MinMaxScaler()
    scaledInputs = scalerIn.fit_transform(combinedInputs.to_numpy())

    fibre_variance = np.ones_like(scaledInputs) * 0

    inputs_tensor = torch.tensor(scaledInputs).to(device)
    variance_tensor = torch.tensor(fibre_variance).to(device)

    inputsZipped = torch.stack((inputs_tensor, (variance_tensor)), dim=1)

    if uncertain:
        edfa1_outputs, edfa1_variances = EDFA_ModelLoad.make_predictions(model, likelihood, inputsZipped, True, "1")
    else:
        edfa1_outputs, edfa1_variances = EDFA_ModelLoad.make_predictions(model, likelihood, inputs_tensor, False, "1")


    # FIBRE 1 MODEL
    offChannelLoss, onChannelLoss, offChannelStd, onChannelStd = Fibre_ModelLoad.getLosses("fibre 1")

    fibre_output, fibre_std = Fibre_ModelLoad.applyLosses(perms.iloc[:len(edfa1_outputs)], offChannelLoss, onChannelLoss, offChannelStd, onChannelStd, edfa1_outputs)
    fibre_variance = Fibre_ModelLoad.combiningUncertainty(edfa1_variances, fibre_std)

    fibre_output = fibre_output * onChannels[:len(fibre_output)]
    fibre_variance = fibre_variance * onChannels[:len(fibre_output)]

    fibre_output[fibre_output == 0] = fibre_output.min().min() - 5


    # EDFA 2 MODEL
    model, likelihood = EDFA_ModelLoad.load_model("2", 18, uncertain)

    
    fibre_output_df = pd.DataFrame(fibre_output)
    # fibre_output_df[fibre_output_df < -50] = -60
    combinedInputs = pd.concat([fibre_output_df, settings[["0", "1"]].iloc[:len(fibre_output_df)]], axis=1)

    fibre_variance = pd.DataFrame(fibre_variance)
    fibre_variance["16"], fibre_variance["17"] = 0,0
    fibre_variance = fibre_variance.to_numpy()

    scalerIn = MinMaxScaler()
    scaledInputs = scalerIn.fit_transform(combinedInputs.to_numpy())

    inputs_tensor = torch.tensor(scaledInputs).to(device)
    variance_tensor = torch.tensor(fibre_variance).to(device)

    inputsZipped = torch.stack((inputs_tensor, (variance_tensor)), dim=1)

    if uncertain:
        edfa2_outputs, edfa2_variances = EDFA_ModelLoad.make_predictions(model, likelihood, inputsZipped, True, "2")
    else:
        edfa2_outputs, edfa2_variances = EDFA_ModelLoad.make_predictions(model, likelihood, inputs_tensor, False, "2")

    # FIBRE 2 MODEL
    offChannelLoss, onChannelLoss, offChannelStd, onChannelStd = Fibre_ModelLoad.getLosses("fibre 2")

    fibre_output, fibre_std = Fibre_ModelLoad.applyLosses(perms.iloc[:len(edfa2_outputs)], offChannelLoss, onChannelLoss, offChannelStd, onChannelStd, edfa2_outputs)
    fibre_variance = Fibre_ModelLoad.combiningUncertainty(edfa2_variances, fibre_std)

    fibre_output = fibre_output * onChannels[:len(fibre_output)]
    fibre_variance = fibre_variance * onChannels[:len(fibre_output)]

    fibre_output[fibre_output == 0] = fibre_output.min().min() - 5


    # EDFA 3 MODEL
    model, likelihood = EDFA_ModelLoad.load_model("3", 19, uncertain)


    fibre_output_df = pd.DataFrame(fibre_output)
    combinedInputs = pd.concat([fibre_output_df, settings[["0", "1", "2"]].iloc[:len(fibre_output_df)]], axis=1)

    fibre_variance = pd.DataFrame(fibre_variance)
    fibre_variance["16"] = 0
    fibre_variance["17"] = 0
    fibre_variance["18"] = 0
    fibre_variance = fibre_variance.to_numpy()

    scalerIn = MinMaxScaler()
    scaledInputs = scalerIn.fit_transform(combinedInputs.to_numpy())

    inputs_tensor = torch.tensor(scaledInputs).to(device)
    variance_tensor = torch.tensor(fibre_variance).to(device)

    inputsZipped = torch.stack((inputs_tensor, (variance_tensor)), dim=1)

    if uncertain:
        edfa3_outputs, edfa3_variances = EDFA_ModelLoad.make_predictions(model, likelihood, inputsZipped, True, "3")
    else:
        edfa3_outputs, edfa3_variances = EDFA_ModelLoad.make_predictions(model, likelihood, inputs_tensor, False, "3")


    # FIBRE 3 MODEL
    offChannelLoss, onChannelLoss, offChannelStd, onChannelStd = Fibre_ModelLoad.getLosses("fibre 3")

    fibre_output, fibre_std = Fibre_ModelLoad.applyLosses(perms.iloc[:len(edfa3_outputs)], offChannelLoss, onChannelLoss, offChannelStd, onChannelStd, edfa3_outputs)
    fibre_variance = Fibre_ModelLoad.combiningUncertainty(edfa3_variances, fibre_std)

    fibre_output = fibre_output * onChannels[:len(fibre_output)]
    fibre_variance = fibre_variance * onChannels[:len(fibre_output)]

    fibre_output[fibre_output == 0] = fibre_output.min().min() - 5



    # EDFA 4 MODEL
    model, likelihood = EDFA_ModelLoad.load_model("4", 20, uncertain)

    fibre_output_df = pd.DataFrame(fibre_output)
    combinedInputs = pd.concat([fibre_output_df, settings[["0", "1", "2", "3"]].iloc[:len(fibre_output_df)]], axis=1)

    fibre_variance = pd.DataFrame(fibre_variance)
    fibre_variance["16"] = 0
    fibre_variance["17"] = 0
    fibre_variance["18"] = 0
    fibre_variance["19"] = 0
    fibre_variance = fibre_variance.to_numpy()

    scalerIn = MinMaxScaler()
    scaledInputs = scalerIn.fit_transform(combinedInputs.to_numpy())

    inputs_tensor = torch.tensor(scaledInputs).to(device)
    variance_tensor = torch.tensor(fibre_variance).to(device)

    inputsZipped = torch.stack((inputs_tensor, (variance_tensor)), dim=1)

    if uncertain:
        edfa4_outputs, edfa4_variances = EDFA_ModelLoad.make_predictions(model, likelihood, inputsZipped, True, "4")
    else:
        edfa4_outputs, edfa4_variances = EDFA_ModelLoad.make_predictions(model, likelihood, inputs_tensor, False, "4")


    # FIBRE 4 MODEL
    offChannelLoss, onChannelLoss, offChannelStd, onChannelStd = Fibre_ModelLoad.getLosses("fibre 4")

    fibre_output, fibre_std = Fibre_ModelLoad.applyLosses(perms.iloc[:len(edfa4_outputs)], offChannelLoss, onChannelLoss, offChannelStd, onChannelStd, edfa4_outputs)
    fibre_variance = Fibre_ModelLoad.combiningUncertainty(edfa4_variances, fibre_std)


    fibre_output = fibre_output * onChannels[:len(fibre_output)]
    fibre_variance = fibre_variance * onChannels[:len(fibre_output)]

    fibre_output[fibre_output == 0] = fibre_output.min().min() - 5

    # EDFA 5 MODEL
    model, likelihood = EDFA_ModelLoad.load_model("5", 21, uncertain)

    fibre_output_df = pd.DataFrame(fibre_output)
    combinedInputs = pd.concat([fibre_output_df, settings[["0", "1", "2", "3", "4"]].iloc[:len(fibre_output_df)]], axis=1)

    fibre_variance = pd.DataFrame(fibre_variance)
    fibre_variance["16"] = 0
    fibre_variance["17"] = 0
    fibre_variance["18"] = 0
    fibre_variance["19"] = 0
    fibre_variance["20"] = 0
    fibre_variance = fibre_variance.to_numpy()

    scalerIn = MinMaxScaler()
    scaledInputs = scalerIn.fit_transform(combinedInputs.to_numpy())

    inputs_tensor = torch.tensor(scaledInputs).to(device)
    variance_tensor = torch.tensor(fibre_variance).to(device)

    inputsZipped = torch.stack((inputs_tensor, (variance_tensor)), dim=1)
    if uncertain:
        edfa5_outputs, edfa5_variances = EDFA_ModelLoad.make_predictions(model, likelihood, inputsZipped, True, "5")
    else:
        edfa5_outputs, edfa5_variances = EDFA_ModelLoad.make_predictions(model, likelihood, inputs_tensor, False, "5")


    # FIBRE 5 MODEL
    offChannelLoss, onChannelLoss, offChannelStd, onChannelStd = Fibre_ModelLoad.getLosses("fibre 5")

    fibre_output, fibre_std = Fibre_ModelLoad.applyLosses(perms.iloc[:len(edfa5_outputs)], offChannelLoss, onChannelLoss, offChannelStd, onChannelStd, edfa5_outputs)
    fibre_variance = Fibre_ModelLoad.combiningUncertainty(edfa5_variances, fibre_std)


    fibre_output = fibre_output * onChannels[:len(fibre_output)]
    fibre_variance = fibre_variance * onChannels[:len(fibre_output)]

    fibre_output[fibre_output == 0] = fibre_output.min().min() - 5



    fibre_output_df = pd.DataFrame(fibre_output)
    combinedInputs = pd.concat([fibre_output_df, settings.iloc[:len(fibre_output_df)]], axis=1)

    fibre_variance = pd.DataFrame(fibre_variance)
    fibre_variance["16"] = 0
    fibre_variance["17"] = 0
    fibre_variance["18"] = 0
    fibre_variance["19"] = 0
    fibre_variance["20"] = 0
    fibre_variance["21"] = 0
    fibre_variance = fibre_variance.to_numpy()

    scalerIn = MinMaxScaler()
    scaledInputs = scalerIn.fit_transform(combinedInputs.to_numpy())

    outputsArr = outputPowers.to_numpy()

    scaledVariance = fibre_variance

    testIndex = 300

    X_train, y_train = scaledInputs[:numTraining], outputsArr[:numTraining]
    X_test, y_test = scaledInputs[testIndex:], outputsArr[testIndex:]

    X_train, y_train = torch.tensor(X_train).to(device), torch.tensor(y_train).to(device)
    X_test, y_test = torch.tensor(X_test).to(device), torch.tensor(y_test).to(device)

    # Variance only
    train_std, test_std = scaledVariance[:numTraining], scaledVariance[numTraining:]
    train_std, test_std = torch.tensor(train_std).to(device), torch.tensor(test_std).to(device)

    inputsZipped = torch.stack((X_train, (train_std)), dim=1)
    TestingZipped = torch.stack((X_test, (test_std)), dim=1)


    class ExactGPModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super().__init__(train_x, train_y, likelihood)
            if uncertain:
                self.mean_module = UncertainMean.ConstantMean()
                self.covar_module = (gpytorch.kernels.ScaleKernel(LKUncertain.LaplacianKernel()) + gpytorch.kernels.ScaleKernel(UncertainKernel.UncertainKernel(ard_num_dims=22))).cuda()
            else:
                self.mean_module = gpytorch.means.ConstantMean()
                self.covar_module = (gpytorch.kernels.ScaleKernel(LK.LaplacianKernel()) + gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=22))).cuda()

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

    path = "models/"
    if uncertain:
        torch.save(model.state_dict(), path+"nigp/EDFA6.pth")
        torch.save(train_data, path+"nigp/EDFA6_trainingData.pt")
    else:
        torch.save(model.state_dict(), path+"gp/EDFA6.pth")
        torch.save(train_data, path+"gp/EDFA6_trainingData.pt")
    
    return error