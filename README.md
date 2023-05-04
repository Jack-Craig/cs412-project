# CS412 Final Project
Rohan Chaturvedula and Jack Craig


## How to run:
1. Setup `.env` file
1. `pip3 install -r requirements.txt`
1. `python3 evaluator.py`

## .env format
`APCA_API_KEY_ID=` Key ID from Alpaca API

`APCA_API_SECRET_KEY=` Secret key from Alpaca API

`APCA_API_BASE_URL=` Base url (probably paper) from Alpaca API

## Data
We use data from the Alpaca trade API. This is an API used in algotrading that alows algorithmic execution of trades and historical stock prices. We do not make use of its trade execution capabilities in this project. 

To access this API you will need to create a free account at https://alpaca.markets/. This gives you access to the API key required for accessing historical data.