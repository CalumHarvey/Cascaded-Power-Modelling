from torch import nn
import torch
import pandas as pd
import numpy as np
import gpytorch
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import CustomKernels.LaplacianKernel as LK
from gpytorch.mlls import SumMarginalLogLikelihood

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def trainModel(inputPowers, outputPowers, settings, perms, numTraining=300):

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



    X_train, y_train = scaledInputs[:numTraining], outputPowersArr[:numTraining]
    X_test, y_test = scaledInputs[numTraining:], outputPowersArr[numTraining:]

    X_train, y_train = torch.tensor(X_train).to(device), torch.tensor(y_train).to(device)
    X_test, y_test = torch.tensor(X_test).to(device), torch.tensor(y_test).to(device)


    class ExactGPModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super().__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = (gpytorch.kernels.ScaleKernel(LK.LaplacianKernel()) + gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=22))).cuda()

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        

    likelihoods = []
    models = []
    train_data = [(X_train, y_train[:,0]), (X_train, y_train[:,1]), (X_train, y_train[:,2]), (X_train, y_train[:,3]), (X_train, y_train[:,4]), (X_train, y_train[:,5]), (X_train, y_train[:,6]), (X_train, y_train[:,7]), (X_train, y_train[:,8]), (X_train, y_train[:,9]), (X_train, y_train[:,10]), (X_train, y_train[:,11]), (X_train, y_train[:,12]), (X_train, y_train[:,13]), (X_train, y_train[:,14]), (X_train, y_train[:,15])]

    for i in range(16):
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = ExactGPModel(train_data[i][0], train_data[i][1], likelihood)
        likelihoods.append(likelihood)
        models.append(model)

    model = gpytorch.models.IndependentModelList(*models).to(device)
    likelihood = gpytorch.likelihoods.LikelihoodList(*[model.likelihood for model in models]).to(device)

    mll = SumMarginalLogLikelihood(likelihood, model)


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

    allPreds = []
    lowers = []
    uppers = []
    variances = []

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
    activeChannels = perms + 1

    activeChannels[activeChannels > 0 ] = 1
    activeChannels[activeChannels == 0] = None

    error = np.nanmean(np.abs((y_test.cpu().numpy()[:len(predsFormatted)] - predsFormatted) * activeChannels[:len(predsFormatted)]))

    print(error)
    return error