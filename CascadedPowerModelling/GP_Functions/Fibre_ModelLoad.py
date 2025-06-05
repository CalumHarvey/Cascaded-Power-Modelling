from torch import nn
import mysql.connector
import torch
import pandas as pd
import numpy as np
import gpytorch
from sklearn.preprocessing import MinMaxScaler
from tqdm.notebook import tqdm
import CustomKernels.LaplacianKernel as LK
import random

from GP_Functions import EDFA_ModelLoad

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def getLosses(folderName):
    path = "CascadedPowerModelling/data/"
    inputPowers = pd.read_csv(path+folderName+"/Inputs.csv", index_col=0)
    outputPowers = pd.read_csv(path+folderName+"/Outputs.csv", index_col=0)
    perms = pd.read_csv(path+folderName+"/perms.csv", index_col=0)

    inputPowersArr = inputPowers.to_numpy()
    outputPowersArr = outputPowers.to_numpy()

    """
    Off Channels
    """
    onChannels = []

    for x in np.array(perms).flatten():
        if x == -1:
            onChannels.append(1)
        else:
            onChannels.append(0)

    onChannels = np.array(onChannels).reshape(len(perms),16)

    loss = inputPowersArr - outputPowersArr
    loss = loss * onChannels

    loss[loss == 0] = None

    averageLossOffChannels = []
    stdLossOffChannels = []

    for c in range(16):
        averageLossOffChannels.append(np.nanmean(loss[:,c]))
        stdLossOffChannels.append(np.nanstd(loss[:,c]))


    """
    On Channels
    """
    onChannels = []

    for x in np.array(perms).flatten():
        if x == -1:
            onChannels.append(0)
        else:
            onChannels.append(1)

    onChannels = np.array(onChannels).reshape(len(perms),16)

    loss = inputPowersArr - outputPowersArr
    loss = loss * onChannels
    loss[loss < 1] = None

    averageLossOnChannels = []
    stdLossOnChannels = []

    for c in range(16):
        averageLossOnChannels.append(np.nanmean(loss[:,c]))
        stdLossOnChannels.append(np.nanstd(loss[:,c]))

    return averageLossOffChannels, averageLossOnChannels, stdLossOffChannels, stdLossOnChannels


def applyLosses(perms, lossOffChannels, lossOnChannels, offChannelStd, onChannelStd, data):

    """
    Off Channels
    """

    offChannels = []

    for x in np.array(perms).flatten():
        if x == -1:
            offChannels.append(1)
        else:
            offChannels.append(0)

    offChannels = np.array(offChannels).reshape(len(perms),16)

    predOutputOffChannels = (data - lossOffChannels) * offChannels
    predOffChannelStd = offChannelStd * offChannels


    """
    On Channels
    """

    onChannels = []

    for x in np.array(perms).flatten():
        if x == -1:
            onChannels.append(0)
        else:
            onChannels.append(1)

    onChannels = np.array(onChannels).reshape(len(perms),16)

    predOutputOnChannels = (data - lossOnChannels) * onChannels
    predOnChannelStd = onChannelStd * onChannels

    """
    Combining on and off channels
    """

    predOutputOnChannels = predOutputOnChannels.flatten()
    predOutputOffChannels = predOutputOffChannels.flatten()

    predOffChannelStd = predOffChannelStd.flatten()
    predOnChannelStd = predOnChannelStd.flatten()

    combinedPredsPowers = []
    combinedPredsStds = []

    for x in range(len(predOutputOffChannels)):
        if predOutputOnChannels[x] != 0:
            combinedPredsPowers.append(predOutputOnChannels[x])
            combinedPredsStds.append(predOnChannelStd[x])
        else:
            combinedPredsPowers.append(predOutputOffChannels[x])
            combinedPredsStds.append(predOffChannelStd[x])

    combinedPredsPowers = np.array(combinedPredsPowers).reshape(len(perms),16)
    combinedPredsStds = np.array(combinedPredsStds).reshape(len(perms),16)


    return combinedPredsPowers, combinedPredsStds


def combiningUncertainty(edfaVar, fibreVar):
    return np.mean(edfaVar, axis=0) + fibreVar