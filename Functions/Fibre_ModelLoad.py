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

from Functions import EDFA_Modelling

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def getLosses(folderName):
    inputPowers = pd.read_csv("data/"+folderName+"/Inputs.csv", index_col=0)
    outputPowers = pd.read_csv("data/"+folderName+"/Outputs.csv", index_col=0)
    perms = pd.read_csv("data/"+folderName+"/perms.csv", index_col=0)

    inputPowersArr = inputPowers.to_numpy()
    outputPowersArr = outputPowers.to_numpy()

    onChannels = perms.copy()
    onChannels[onChannels > -1] = 1
    onChannels[onChannels == -1] = None

    onChannels = onChannels.to_numpy()


    """
    On Channels
    """

    loss = inputPowersArr - outputPowersArr
    loss = loss * onChannels

    averageLossOnChannels = []
    stdLossOnChannels = []

    for c in range(16):
        averageLossOnChannels.append(np.nanmean(loss[:,c]))
        stdLossOnChannels.append(np.nanstd(loss[:,c]))

    return averageLossOnChannels, stdLossOnChannels




def applyLosses(onChannels, lossOnChannels, onChannelStd, data):

    """
    Off Channels
    """

    # offChannels = []

    # for x in np.array(perms).flatten():
    #     if x == -1:
    #         offChannels.append(1)
    #     else:
    #         offChannels.append(0)

    # offChannels = np.array(offChannels).reshape(len(perms),16)

    # predOutputOffChannels = (data - lossOffChannels) * offChannels
    # predOffChannelStd = offChannelStd * offChannels


    """
    On Channels
    """

    # onChannels = []

    # for x in np.array(perms).flatten():
    #     if x == -1:
    #         onChannels.append(0)
    #     else:
    #         onChannels.append(1)

    # onChannels = np.array(onChannels).reshape(len(perms),16)

    predOutputOnChannels = (data - lossOnChannels) * onChannels
    predOnChannelStd = onChannelStd * onChannels

    # """
    # Combining on and off channels
    # """

    # predOutputOnChannels = predOutputOnChannels.flatten()
    # predOutputOffChannels = predOutputOffChannels.flatten()

    # predOffChannelStd = predOffChannelStd.flatten()
    # predOnChannelStd = predOnChannelStd.flatten()

    # combinedPredsPowers = []
    # combinedPredsStds = []

    # for x in range(len(predOutputOffChannels)):
    #     if predOutputOnChannels[x] != 0:
    #         combinedPredsPowers.append(predOutputOnChannels[x])
    #         combinedPredsStds.append(predOnChannelStd[x])
    #     else:
    #         combinedPredsPowers.append(predOutputOffChannels[x])
    #         combinedPredsStds.append(predOffChannelStd[x])

    # combinedPredsPowers = np.array(combinedPredsPowers).reshape(len(perms),16)
    # combinedPredsStds = np.array(combinedPredsStds).reshape(len(perms),16)


    return predOutputOnChannels, predOnChannelStd


def fibrePredictions(folderName, onChannels, edfa11_outputs, edfa11_variances):

    onChannelLoss, onChannelStd = getLosses(folderName)

    fibre_output, fibre_std = applyLosses(onChannels[:len(edfa11_outputs)], onChannelLoss, onChannelStd, edfa11_outputs)
    fibre_variance = np.median(edfa11_variances, axis=0) + fibre_std

    fibre_output = fibre_output * onChannels[:len(fibre_output)]
    fibre_variance = fibre_variance * onChannels[:len(fibre_output)]

    fibre_output[fibre_output == 0] = fibre_output.min().min() - 5

    return fibre_output, fibre_variance