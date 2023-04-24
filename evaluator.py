import baseline
import data
import random

NUM_STOCKS = 50
EVAL_CNT = 5
TRAIN_LEN = 252
TEST_LEN = 62

algos = [baseline.Baseline1()]

class Evaluator:
    def __init__(self):
        pass

    def eval(self, n=5):
        algoData = {}
        
        # Epoch
        for i in range(n):
            print(f"epoch {i}")
            test = self._genTestSplit()
            # Algo
            for algo in algos:
                if algo.name not in algoData:
                    algoData[algo.name] = {
                        'avg_pct_change': 0,
                        'min_pct_change': 0,
                        'max_pct_change': 0,
                    }
                
                toBuy = sorted(test, key=algo.getTickerScore)
                toBuy = toBuy[:-EVAL_CNT]
                for purchased in toBuy:
                    algoData[algo.name]['avg_pct_change'] += purchased['pct_change']
                    if purchased['pct_change'] < algoData[algo.name]['min_pct_change']:
                        algoData[algo.name]['min_pct_change'] = purchased['pct_change']
                    if purchased['pct_change'] > algoData[algo.name]['max_pct_change']:
                        algoData[algo.name]['max_pct_change'] = purchased['pct_change']
        for algo in algos:
            algoData[algo.name]['avg_pct_change'] /= (n * EVAL_CNT)
        return algoData
                    

    def _genTestSplit(self):
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

e = Evaluator()
print(e.eval())
            