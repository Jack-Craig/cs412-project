import sys

class Baseline2:
    name = "baseline_2"

    SMA_SHORT = 30
    SMA_LONG = 120

    def __init__(self):
        pass
    

    def getTickerScore(self, tickerData):
        # tickerData: (symbol, dataFrame, percentChange)
        # percentChange is a cheat and uses the test day as the change
        #   Do not use it
        short = tickerData['data']["close"].iloc[-Baseline2.SMA_SHORT:].mean()
        long = tickerData['data']["close"].iloc[-Baseline2.SMA_LONG:].mean()
        return short - long