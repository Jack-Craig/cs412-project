class Baseline1:
    name = "baseline_1"
    def __init__(self):
        pass

    # Should return 
    def getTickerScore(self, tickerData):
        return tickerData['pct_change']
