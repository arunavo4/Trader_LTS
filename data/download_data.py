import time
from glob import glob
import pandas as pd
from utils import FileIO
from data import api_v2

config_path = 'config/config_data.yml'
config = FileIO.read_yaml(config_path)
attr = ['date', 'open', 'high', 'low', 'close']

# ticker_list = ['ADT']
# A list of all S&P 500 Companies
company_list = pd.read_csv('us_mkt/companies_list.csv')
ticker_list = list(company_list['Symbol'])

n = 5
api = api_v2.Tiingo(config)

_train_dir = 'us_mkt/Train_data'
# Due to an issue with the Api,
_files = glob(_train_dir + '/*.csv')

done_tickers = [x.split('/')[-1].split('.')[0] for x in _files]

left_tickers = list(set(ticker_list) - set(done_tickers))

start = time.time()
new_ticker_list = [left_tickers[i * n:(i + 1) * n] for i in range((len(left_tickers) + n - 1) // n)]
for ticker_chunk in new_ticker_list:
    hist_df = api.fetch(ticker_chunk, attr)
    for ticker in ticker_chunk:
        try:
            FileIO.save_csv(hist_df[ticker], ticker, _train_dir)
            print("Saved: ", ticker)
        except:
            pass
end = time.time()
print('Asynchronous processing time: {time}s.'.format(time=end - start))
