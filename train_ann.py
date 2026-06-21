from ann import EDFA1Model, EDFA2Model, EDFA3Model, EDFA4Model, EDFA5Model, EDFA6Model, EndToEnd
import pandas as pd
import warnings
warnings.filterwarnings("ignore") # Model loading seen as unsafe by PyTorch but the files are saved and loaded by me, there is no safety issue.

def trainAllModels():
    """ Trains all models for cascaded learning in order from EDFA 1 to EDFA 6
    """

    functionNames = [EDFA1Model.trainModel, EDFA2Model.trainModel, EDFA3Model.trainModel, EDFA4Model.trainModel, EDFA5Model.trainModel, EDFA6Model.trainModel]
    allErrors = []

    for x in range(6):

        path = "data/EDFA "
        inputPowers = pd.read_csv(path+str(x+1)+"/Inputs.csv", index_col=0)
        outputPowers = pd.read_csv(path+str(x+1)+"/Outputs.csv", index_col=0)

        settings = pd.read_csv(path+str(x+1)+"/EDFA_setting.csv", index_col=0)
        perms = pd.read_csv(path+str(x+1)+"/perms.csv", index_col=0)   

        error = functionNames[x](inputPowers, outputPowers, settings, perms)
        print(error)
        allErrors.append(error)

    return allErrors


def trainSingleEDFAModel(x):
    """Trains only a single EDFA model

    Args:
        x (int): Index of EDFA to train from 1 to 6
    """

    functionNames = [EDFA1Model.trainModel, EDFA2Model.trainModel, EDFA3Model.trainModel, EDFA4Model.trainModel, EDFA5Model.trainModel, EDFA6Model.trainModel]

    path = "data/EDFA "
    inputPowers = pd.read_csv(path+str(x)+"/Inputs.csv", index_col=0)
    outputPowers = pd.read_csv(path+str(x)+"/Outputs.csv", index_col=0)

    settings = pd.read_csv(path+str(x)+"/EDFA_setting.csv", index_col=0)
    perms = pd.read_csv(path+str(x)+"/perms.csv", index_col=0)      

    error = functionNames[x](inputPowers, outputPowers, settings, perms)
    print(error)

    return error



def trainEndToEnd():
    """Train End to End GP model
    """

    path = "data/EDFA "
    x = 5 # EDFA 6 is entire link measurements
    inputPowers = pd.read_csv(path+str(x+1)+"/Inputs.csv", index_col=0)
    outputPowers = pd.read_csv(path+str(x+1)+"/Outputs.csv", index_col=0)

    settings = pd.read_csv(path+str(x+1)+"/EDFA_setting.csv", index_col=0)
    perms = pd.read_csv(path+str(x+1)+"/perms.csv", index_col=0)      

    error = EndToEnd.trainModel(inputPowers, outputPowers, settings, perms)
    print(error)

    return error


if __name__ == "__main__":
    # error = trainSingleEDFAModel(0, True)
    # print(error)

    # Standard cascade
    errors = trainAllModels()
    print(errors)

    # End to end model
    # error = trainEndToEnd()
    # print(error)


    pass