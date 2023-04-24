import alpaca_trade_api as apa
from dotenv import load_dotenv
import os
import pandas as pd

DF_LEN = 800

load_dotenv()

api = apa.rest.REST()

def LoadAndSaveTicker(ticker, startDate="2018-01-01", endDate="2023-01-01", override=False):
    if not override and os.path.isfile(f"data/{ticker}.csv"):
        return pd.read_csv(f"data/{ticker}.csv")
    try:
        bars_df = api.get_bars(ticker, '1Day', startDate, endDate, adjustment='raw').df
    except:
        return None
    if len(bars_df) < DF_LEN:
        return None
    bars_df = bars_df.head(DF_LEN)
    bars_df.to_csv(f"data/{ticker}.csv")
    return bars_df

def LoadAllNYSETickers():
    ass = api.list_assets(asset_class="us_equity")
    symbols = []
    for a in ass:
        if a.exchange == "NYSE" and '.' not in a.symbol:
            symbols.append(a.symbol)
    return symbols