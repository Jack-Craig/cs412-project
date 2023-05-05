import sys
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import Train_NN
import torch
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

class Baseline3:
    name = "baseline_3"

    SMA_SHORT = 30
    SMA_LONG = 120

    def __init__(self):
        pass
    
    def getTickerScore(self, tickerData):
        # tickerData: (symbol, dataFrame, percentChange)
        # percentChange is a cheat and uses the test day as the change
        #   Do not use it
        #print(tickerData)
        #print("hello")
        model = Train_NN.Train_NN()
        model.load_state_dict(torch.load("Original_NN_Predictor.pth"))
        model.eval()
        #print("hello")

        features = ["open","high","low","close","volume","trade_count","vwap"]
        #print(tickerData["data"])
        pca = PCA(n_components=2)
        #print(x)
        training_data = []
        input_data = []
        for f in features:
            if max(tickerData["data"][f]) == 0:
                continue  
            tickerData["data"][f] = (tickerData["data"][f] - min(tickerData["data"][f]))/max(tickerData["data"][f])
    # print(y)
        x = tickerData["data"].loc[:, features].values
        #print(x)

        principalComponents = pca.fit(x)
        for i in x:
            x1,x2 = 0,0 
            for j in range(len(i)): 
                #print(principalComponents.components_[0][j], i[j])
                x1 += (principalComponents.components_[0][j] * i[j])
                x2 += (principalComponents.components_[1][j] * i[j])
            input_data.append(x1)
            input_data.append(x2)
        #training_data.append(input_data)
        #print(input)
        return -1 *model(torch.FloatTensor(input_data)).item()