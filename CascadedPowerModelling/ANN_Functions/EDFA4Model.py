from ANN_Functions import Fibre_ModelLoad
from torch import nn
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from torch.utils.data import DataLoader
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, samples, labels):
      'Initialization'
      self.labels = labels
      self.samples = samples

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.samples)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Load data and get label
        X = self.samples[index]
        y = self.labels[index]

        return X, y
  


class NeuralNetwork(nn.Module):
    def __init__(self, input_dims):
        super(NeuralNetwork, self).__init__()

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_dims, 49),
            nn.Linear(49, 49),
            nn.ELU(),
            nn.Linear(49, 49),
            nn.Linear(49, 16),
            )
        
    def forward(self, x):
        output = self.linear_relu_stack(x)
        return output



def trainModel(inputPowers, outputPowers, settings, perms, numTraining=300):

    onChannels = perms.copy()

    onChannels[onChannels > -1] = 1
    onChannels[onChannels == -1] = 0

    outputPowers = outputPowers * onChannels
    inputPowers = inputPowers * onChannels

    inputPowers[inputPowers == 0] = inputPowers.min().min() - 5
    outputPowers[outputPowers == 0] = outputPowers.min().min() - 5

    combinedInputs = pd.concat([inputPowers, settings["0"]], axis=1)

    inputPowersArr = combinedInputs.to_numpy()
    
    scalerIn = MinMaxScaler()
    scaledInputs = scalerIn.fit_transform(inputPowersArr)

    # EDFA 1
    EDFAID = "1"
    path = "CascadedPowerModelling/SavedModels/ANN/"
    model = NeuralNetwork(17).to(device)
    model.load_state_dict(torch.load(path+"EDFA"+EDFAID+".pth", weights_only=True))

    inputs_tensor = torch.tensor(scaledInputs, dtype=torch.float32).to(device)
    model.eval()
    edfa1_outputs = model(inputs_tensor)

    # FIBRE 1
    offChannelLoss, onChannelLoss, offChannelStd, onChannelStd = Fibre_ModelLoad.getLosses("fibre 1")
    fibre_output, _ = Fibre_ModelLoad.applyLosses(perms, offChannelLoss, onChannelLoss, offChannelStd, onChannelStd, edfa1_outputs.detach().cpu().numpy())

    fibre_output = fibre_output * onChannels[:len(fibre_output)]
    fibre_output[fibre_output == 0] = fibre_output.min().min() - 5


    # EDFA 2
    EDFAID = "2"
    path = "CascadedPowerModelling/SavedModels/ANN/"
    model = NeuralNetwork(18).to(device)
    model.load_state_dict(torch.load(path+"EDFA"+EDFAID+".pth", weights_only=True))

    fibre_output_df = pd.DataFrame(fibre_output)
    combinedInputs = pd.concat([fibre_output_df, settings[["0", "1"]]], axis=1)

    scalerIn = MinMaxScaler()
    scaledInputs = scalerIn.fit_transform(combinedInputs.to_numpy())
    inputs_tensor = torch.tensor(scaledInputs, dtype=torch.float32).to(device)

    model.eval()
    edfa2_outputs = model(inputs_tensor)


    # FIBRE 2
    offChannelLoss, onChannelLoss, offChannelStd, onChannelStd = Fibre_ModelLoad.getLosses("fibre 2")
    fibre_output, _ = Fibre_ModelLoad.applyLosses(perms, offChannelLoss, onChannelLoss, offChannelStd, onChannelStd, edfa2_outputs.detach().cpu().numpy())

    fibre_output = fibre_output * onChannels[:len(fibre_output)]
    fibre_output[fibre_output == 0] = fibre_output.min().min() - 5



    # EDFA 3
    EDFAID = "3"
    path = "CascadedPowerModelling/SavedModels/ANN/"
    model = NeuralNetwork(19).to(device)
    model.load_state_dict(torch.load(path+"EDFA"+EDFAID+".pth", weights_only=True))

    fibre_output_df = pd.DataFrame(fibre_output)
    combinedInputs = pd.concat([fibre_output_df, settings[["0", "1", "2"]]], axis=1)

    scalerIn = MinMaxScaler()
    scaledInputs = scalerIn.fit_transform(combinedInputs.to_numpy())

    inputs_tensor = torch.tensor(scaledInputs, dtype=torch.float32).to(device)

    model.eval()
    edfa3_outputs = model(inputs_tensor)


    # FIBRE 3
    offChannelLoss, onChannelLoss, offChannelStd, onChannelStd = Fibre_ModelLoad.getLosses("fibre 3")
    fibre_output, _ = Fibre_ModelLoad.applyLosses(perms, offChannelLoss, onChannelLoss, offChannelStd, onChannelStd, edfa3_outputs.detach().cpu().numpy())

    fibre_output = fibre_output * onChannels[:len(fibre_output)]
    fibre_output[fibre_output == 0] = fibre_output.min().min() - 5



    # EDFA 4
    fibre_output_df = pd.DataFrame(fibre_output)
    combinedInputs = pd.concat([fibre_output_df, settings], axis=1)

    scalerIn = MinMaxScaler()
    scaledInputs = scalerIn.fit_transform(combinedInputs.to_numpy())

    outputPowersArr = outputPowers.to_numpy()


    
    X_train, y_train = scaledInputs[:numTraining], outputPowersArr[:numTraining]
    X_test, y_test = scaledInputs[numTraining:], outputPowersArr[numTraining:]


    training_data = Dataset(X_train, y_train)
    test_data = Dataset(X_test, y_test)

    # Create Dataloaders
    train_dataloader = DataLoader(training_data, batch_size=128, shuffle=True, num_workers=0, pin_memory=True)
    test_dataloader = DataLoader(test_data, batch_size=128, shuffle=False, num_workers=0, pin_memory=True)

    model = NeuralNetwork(20).to(device)
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=0.00205)

    loss_fn = nn.L1Loss(reduction='mean')

    trainingLosses = []

    n_epoch = 3000

    for epoch in tqdm(range(1, n_epoch+1), ncols=200):
        trainLoss = 0.0
        model.train()
        for x, y in train_dataloader:
            x, y = x.to(device).to(torch.float32), y.to(device).to(torch.float32)
            optimizer.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()
            trainLoss += loss.item()
        trainingLosses.append(trainLoss)

    results = torch.tensor([]).to(device)
    model.eval()
    with torch.no_grad():
        for x, y in test_dataloader:
            x, y = x.to(device).to(torch.float32), y.to(device).to(torch.float32)
            target = model(x)
            results = torch.cat((results, target))

    results = results.cpu().numpy()

    activeChannels = perms + 1
    activeChannels[activeChannels > 0 ] = 1
    activeChannels[activeChannels == 0] = None

    error = np.nanmean(np.abs((y_test[:len(results)] - results) * activeChannels[:len(results)]))

    path = "CascadedPowerModelling/SavedModels/ANN/"
    torch.save(model.state_dict(), path+"EDFA4.pth")

    return error