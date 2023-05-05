import baseline
import sma_cross
import data
import random
import NN
import sys
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


NUM_STOCKS = 50
EVAL_CNT = 5
TRAIN_LEN = 252
TEST_LEN = 62
class Train_NN(nn.Module):

    def __init__(self):
        super(Train_NN, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel

        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(504, 120)  
        self.fc2 = nn.Linear(120, 32)
        self.fc3 = nn.Linear(32, 8)
        self.fc4 = nn.Linear(8, 1)
    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x




def genTestSplit():
    allTickers = data.LoadAllNYSETickers()
    selected = random.sample(allTickers, NUM_STOCKS)
    
    # int to start index from. Assumes stored in descending order
    iStart = random.randint(0, data.DF_LEN - TRAIN_LEN - TEST_LEN)
    iEnd = iStart + TRAIN_LEN
    train = []
    for ticker in selected:
        ticker_data = data.LoadAndSaveTicker(ticker)
        
        if ticker_data is None:
            continue
        checkDay = ticker_data.iloc[iEnd + TEST_LEN]
        ticker_data_trimmed = ticker_data.iloc[iStart : iEnd]
        train.append({
            'symbol': ticker,
            'data': ticker_data_trimmed,
            'pct_change': (checkDay['close']-ticker_data_trimmed.iloc[-1]['close']) / ticker_data_trimmed.iloc[-1]['close']
        })
    return train
    

def train(): 
    
    tickerData = genTestSplit()
    features = ["open","high","low","close","volume","trade_count","vwap"]
    #print(tickerData)
    pca = PCA(n_components=2)
    #print(x)
    training_data = []
    target_data = []
    for y in tickerData:
        input_data = []
        for f in features: 
            y["data"][f] = (y["data"][f] - min(y["data"][f]))/max(y["data"][f])
       # print(y)
        x = y["data"].loc[:, features].values
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
        training_data.append(input_data)
        target_data.append([y["pct_change"]])
    
    #print(len(input_data))
    #print(principalComponents.components_)
    #print(len(input_data))
    

    input, target = [torch.FloatTensor(training_data), torch.FloatTensor(target_data)]
    train = Train_NN()
    optimizer = optim.SGD(train.parameters(), lr=0.001)
    #print(train)
    #print(target)
    for k in range(10000):
        for i in range(len(input)): 
            output = train(input[i])
        # target = torch.randn(10)  # a dummy target, for example
        # target = target.view(1, -1)  # make it the same shape as output
            criterion = nn.MSELoss()
        # create your optimizer
            # in your training loop:
            optimizer.zero_grad()   # zero the gradient buffers
            loss = criterion(output, target[i])
            loss.backward()
            optimizer.step()
            #print(loss.item())
    final = []
    for i in range(len(input)): 
            final.append([train(input[i]).item(),target[i]])

    df = pd.DataFrame(final, columns=['Prediction', "Actual"])
    print(df)

    print(train(input[0]).item(), " Prediction")
    torch.save(train.state_dict(), "Original_NN_Predictor.pth")  
    # print(target[0])

    #print(train.parameters())