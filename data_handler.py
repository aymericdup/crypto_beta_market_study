import pandas as pd
import numpy as np
from os import listdir, path
from typing import Tuple, Optional

import helper

def load_data_from_flat_file(file_path : str) -> pd.DataFrame :
    '''
    Load data from a file path and return a dataframe

    columns loaded: Date,cci30_open,cci30_high,cci30_low,cci30_close,cci30_volume,price_0x,in_index_0x,pric,etc.
    '''
    df = pd.DataFrame()
    try:
        df = pd.read_csv(file_path)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df.sort_index(inplace=True)

    except Exception as ex:
        print(f'Impossible to read: {file_path} due to {ex}')
    finally:
        return df

'''
def extract_market(df : pd.DataFrame) -> pd.DataFrame:
    market = pd.DataFrame()
    try:
        mask_columns_market = [col for col in df.columns if ''.startswith('cci30')]
        market = df[mask_columns_market].copy()
        market.rename(columns={col : col.replace('cci30_', '') for col in mask_columns_market}, inplace=True)
    except Exception as ex: print(f'Impossible to extrat market: {ex}')
    finally: return market

def extract_prices(df : pd.DataFrame) -> pd.DataFrame:
    prices = pd.DataFrame()
    try:
        mask_columns_coins = [col for col in df.columns if ''.startswith('price_')]
        prices = df[mask_columns_coins].copy()
        prices.rename(columns={col : col.replace('price_', '') for col in mask_columns_coins}, inplace=True)
    except Exception as ex: print(f'Impossible to extrat price: {ex}')
    finally: return prices

def extract_constituents(df : pd.DataFrame) -> pd.DataFrame:
    in_index = pd.DataFrame()
    try:
        mask_columns_constituents = [col for col in df.columns if ''.startswith('in_index_')]
        in_index = df[mask_columns_constituents].copy()
        in_index.rename(columns={col : col.replace('in_index_', '') for col in mask_columns_constituents}, inplace=True)
    except Exception as ex: print(f'Impossible to extrat constituents: {ex}')
    finally: return in_index
'''

def extract_features(df : pd.DataFrame, target_col : str) -> pd.DataFrame:
    feature_df = pd.DataFrame()
    try:
        mask_columns_feature = [col for col in df.columns if col.startswith(target_col)]
        feature_df = df[mask_columns_feature].copy()
        feature_df.rename(columns={col : col.replace(target_col, '') for col in mask_columns_feature}, inplace=True)
    except Exception as ex: print(f'Impossible to extract feature of target_col={target_col}: {ex}')
    finally: return feature_df

def compute_returns(df : pd.DataFrame) -> pd.DataFrame :
    '''
    Compute log-returns
    '''
    returns = pd.DataFrame()
    try:
        returns = np.log(df/df.shift(1))
    except Exception as ex: print(f'Impossible to compute returns: {ex}')
    finally: return returns

def mask_out_prices_no_constituents(df : pd.DataFrame, in_index : pd.DataFrame) -> pd.DataFrame:
    '''
    Return a dataframe with 'NaN' value for no-constituent period
    '''
    masked = pd.DataFrame()
    try:
        common = df.columns.intersection(in_index.columns)
        masked = df[common].where(in_index[common])
        
    except Exception as ex : print(f'Impossible to mask out prices from no constituents: {ex}')
    finally: return masked

def load_daily_data(filtering_in_index : bool=False) -> dict:
    return load_data(helper.DAILY_DATA_FILE_PATH, 'daily', filtering_in_index)

def load_weekly_data(filtering_in_index : bool =False) -> dict:
    return load_data(helper.WEEKLY_DATA_FILE_PATH, 'weekly', filtering_in_index)

def load_data(file_path : str, frequency : str, filtering_in_index : bool) -> dict :
    '''
    Load daily data from the dataset provided by cryptounited

    Returns a dictionnary with:
    markets candle
    prices
    returns
    in_index
    '''
    print(f'Load {file_path}, frequency={frequency}')

    # load data from flat file
    df = load_data_from_flat_file(file_path)
    if df.empty: return

    print(f'Extract feature (market, coin_prices, in_index)')

    # extract CCI30 OHLC raws
    market_ohlc = extract_features(df, 'cci30_')
    if market_ohlc.empty: return

    # extract prices for each coin
    coin_prices = extract_features(df, 'price_')
    if coin_prices.empty: return

    # extract an indicatrice for each coin: True for is in index otherwise False
    in_index = extract_features(df, 'in_index_')
    if in_index.empty: return

    # compute returns
    market_c = market_ohlc['close'].copy()
    market_returns = compute_returns(market_c)
    coins_returns = compute_returns(coin_prices)

    # mask-out no constituents
    if filtering_in_index:
        coins_returns = mask_out_prices_no_constituents(coins_returns, in_index)

    # drop first row NaN from return computation
    coins_returns = coins_returns.iloc[1:]
    market_returns = market_returns.iloc[1:]

    return {
        "coin_prices": coin_prices,
        "in_index": in_index,
        "market_close": market_c,
        "market_ohlc": market_ohlc,
        "coins_returns": coins_returns,
        "market_returns": market_returns,
        "freq": frequency,
        "raw_df": df,
    }

def check_integrity(df : pd.DataFrame) -> dict:

    # check for duplicates
    nb_duplicates = df.duplicated().sum()

    # check for NaN
    nb_nan = df.isna().sum().sum()

    return {
        'nb_rows' : len(df),
        'dim' : df.shape,
        'min_date': df.index[0].date(),
        'max_date': df.index[-1].date(),
        'nb_duplicates': nb_duplicates,
        'nb_nan': nb_nan
    }

if __name__ == "__main__":
    
    print(f'Run a test')
    data = load_daily_data()

    if not data: print(f'load daily data failed!')

    coin_prices = data['coin_prices']
    if coin_prices.empty: 
        print(f'Prices loading have been encountered a problem!')
    else:
        print('Check integrity coin_prices')
        check = check_integrity(coin_prices)
        print(check)
        print(coin_prices.describe())

        # bitcoin check
        print(coin_prices['bitcoin'].describe())

        print('Check integrity market_returns')
        check_market = check_integrity(data['market_returns'])
        print(check_market)
        print(data['market_returns'].describe())

        
    
