import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from typing import Optional, Dict, List
import time

import helper
import data_handler

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

def _get_empty_result(idx, columns = ['alpha', 'beta', 'r2', 'se_beta', 'res_vol']) -> pd.DataFrame :
    return pd.DataFrame(index=idx, columns=columns, dtype=float)

def _evalute_ols(y : np.array, x : np.array, beta : np.array) -> dict :
    '''
    Evalute the performance of an OLS, by computing r², se_beta and volatility of residuals

    Returns a dictionary that contains beta / alpha / r_squared / SE_beta / residual_stdev
    '''
    n = len(y)
    alpha = np.mean(y) - beta * np.mean(x)

    # Residuals (computed with close-to-close returns)
    res = y - alpha - beta * x
    ss_res = np.sum(res ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 1e-15 else 0.0

    # SE via Option A: use close-to-close SSX (conservative proxy)
    sigma2_eps = ss_res / (n - 2) if n > 2 else np.nan
    ssx_cc = np.sum((x - np.mean(x)) ** 2)
    se_beta = np.sqrt(sigma2_eps / ssx_cc) if (n > 2 and ssx_cc > 1e-15) else np.nan

    res_vol = np.std(res, ddof=1)
        
    return {'alpha' : alpha, 'beta' : beta, 'r2' : r2, 'se_beta' : se_beta, 'res_vol' : res_vol}

def compute_rolling_ols_beta(coin_returns : pd.Series, market_returns : pd.Series, window : int) -> pd.DataFrame :
    '''
    Compute a rolling OLS over a window between coin returns and market returns

    Returns a dataframe of beta / alpha / r_squared / SE_beta / residual_stdev
    '''

    # build a dataframe which contaneate coin & market
    merged = pd.concat([coin_returns, market_returns], join='inner', axis=1, keys=['coin', 'market']).dropna()

    n = len(merged)
    result = _get_empty_result(merged.index)

    for i in range(window, n):
        k = i - window
        y,x = merged['coin'].iloc[k:i].values,  merged['market'].iloc[k:i].values
        X = sm.add_constant(x)
        
        try:
            res = sm.OLS(y, X).fit()
            result.loc[merged.index[i-1]] = [res.params[0], res.params[1], res.rsquared, res.bse[1], np.std(res.resid, ddof=1)]
        except Exception as ex : 
            print(f'compute_rolling_ols: error during fitting at {k}:{i}: {ex}')
            continue

    return result.dropna(subset=['beta'])

def compute_sw_rolling_beta(coin_returns : pd.Series, market_returns : pd.Series, window : int, delta : int = 3) -> pd.DataFrame :
    '''
    Compute a rolling slope-winsorized OLS over a window between coin returns and market returns. By default, delta = 3 as suggested by Sila (2025)
    Returns a dataframe of beta / alpha / r_squared / SE_beta / residual_stdev
    '''
    # build a dataframe which contaneate coin & market
    merged = pd.concat([coin_returns, market_returns], join='inner', axis=1, keys=['coin', 'market']).dropna()
    n = len(merged)
    result = _get_empty_result(merged.index)

    for i in range(window, n):
        k = i - window
        y,x = merged['coin'].iloc[k:i].values,  merged['market'].iloc[k:i].values

        # estimate the slope winsorized bounds
        # lower bound = rm . (1-delta)
        # upper bound = rm . (1+delta)
        lower_b, upper_b = x * (1 - delta), x * (1 + delta)
        lower_b, upper_b = np.minimum(lower_b, upper_b), np.maximum(lower_b, upper_b) # handle case where rm (x) is negative therefore swaps low/max
        y_sw = np.clip(y, a_min=lower_b, a_max=upper_b)
        X = sm.add_constant(x)
        
        try:
            res = sm.OLS(y_sw, X).fit()
            result.loc[merged.index[i-1]] = [res.params[0], res.params[1], res.rsquared, res.bse[1], np.std(res.resid, ddof=1)]
        except Exception as ex : 
            print(f'compute_sw_rolling_beta: error during fitting at {k}:{i}: {ex}')
            continue

    return result.dropna(subset=['beta'])

def compute_wls_rolling_beta(coin_returns : pd.Series, market_returns : pd.Series, window : int, halflife : int) -> pd.DataFrame :
    '''
    Compute a rolling WLS over a window between coin returns and market returns. Note that the decay is defined by the half-life
    Returns a dataframe of beta / alpha / r_squared / SE_beta / residual_stdev
    '''
    merged = pd.concat([coin_returns, market_returns], join='inner', axis=1, keys=['coin', 'market']).dropna()
    n = len(merged)
    result = _get_empty_result(merged.index)

    for i in range(window, n):
        k = i - window
        y,x = merged['coin'].iloc[k:i].values,  merged['market'].iloc[k:i].values

        # build weight where wk = exp( (-ln(2) / hf) * k)
        m = len(y) # equals to window
        decay = np.log(2) / halflife
        ws = np.exp(-decay * np.arange(m)[::-1]) # reverse due to recent obs matter more

        X = sm.add_constant(x)
        try:
            res = sm.WLS(y, X, weights=ws).fit()
            result.loc[merged.index[i-1]] = [res.params[0], res.params[1], res.rsquared, res.bse[1], np.std(res.resid, ddof=1)]
        except Exception as ex : 
            print(f'compute_sw_rolling_beta: error during fitting at {k}:{i}: {ex}')
            continue
    return result.dropna(subset=['beta'])

def compute_vol_range_rolling_beta(coin_returns : pd.Series, market_returns : pd.Series, markets : pd.DataFrame, window : int) -> pd.DataFrame :
    '''
    Compute a rolling vol range beta over a window between coin returns and market returns. Note vol range means using High/Low for estimating variance (follow Garman-Klass) instead of using close-to-close
    Returns a dataframe of beta / alpha / r_squared / SE_beta / residual_stdev
    '''
    merged = pd.concat([coin_returns.rename('coin'), market_returns.rename('market'), markets[['open', 'high', 'low', 'close']]], join='inner', axis=1).dropna()
    n = len(merged)
    result = _get_empty_result(merged.index)

    def gk_var(open : np.array, high : np.array, low : np.array, close : np.array):
        hl = np.log(high / low)
        co = np.log(close / open)
        return 0.5 * hl ** 2 - (2 * np.log(2) - 1) * co ** 2
    

    for i in range(window, n):
        k = i - window
        selection = merged.iloc[k:i]

        # compute gk values
        gk_vars = gk_var(selection['open'].values, selection['high'].values, selection['low'].values, selection['close']).values
        gk_vars = np.maximum(gk_vars, helper.ZERO_APPROX)
        var_m_gk = np.mean(gk_vars)
        if var_m_gk < helper.ZERO_APPROX : continue # try to avoid case where the var equals 0 -> dividing by zero into beta calculation

        y,x = selection['coin'].values, selection['market'].values

        # lead OLS
        cov_ri_rm = np.cov(x,y, ddof=1)[0, 1]
        beta = cov_ri_rm / var_m_gk

        result_eval = _evalute_ols(y, x, beta)
        '''
        alpha = np.mean(y) - beta * np.mean(x)

        # Residuals (computed with close-to-close returns)
        res = y - alpha - beta * x
        ss_res = np.sum(res ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 1e-15 else 0.0

        # SE via Option A: use close-to-close SSX (conservative proxy)
        sigma2_eps = ss_res / (n - 2) if n > 2 else np.nan
        ssx_cc = np.sum((x - np.mean(x)) ** 2)
        se_beta = np.sqrt(sigma2_eps / ssx_cc) if (n > 2 and ssx_cc > 1e-15) else np.nan

        res_vol = np.std(res, ddof=1)

        result.loc[result.index[i-1]] = [alpha, beta, r2, se_beta, res_vol]
        '''

        result.loc[result.index[i-1]] = [result_eval['alpha'], result_eval['beta'], result_eval['r2'], result_eval['se_beta'], result_eval['res_vol']]
    
    return result.dropna(subset=["beta"])

def compute_vasicek_shrinkage_rolling_beta(coin_returns : pd.Series, market_returns : pd.Series, cs_mean_betas : pd.Series, cs_var_betas : pd.Series, betas : pd.Series, se_betas : pd.Series, window : int) -> pd.DataFrame :
    '''
    Compute a rolling Vasice shrinkage beta over a window between coin returns and market returns
    Note that beta_vck = w.beta_ols + (1-w).beta_prior where beta_prior is the cross-sectional mean of the OLS betas.
    Furthermore, w is described as Var(beta_prior) / [Var(beta_prior) + Var(beta_ols)]

    Parameters:
    cs_mean_betas is the avg beta across all coins which are constituents of the indice at a particular time
    cs_var_betas is the variance of beta across all coins which are constituents of the indice at a particular time
    betas is the rolling ols beta of the coin
    se_betas is the rolling se of the beta of the coin

    Returns a dataframe of beta / alpha / r_squared / SE_beta / residual_stdev
    '''
    aligned = pd.concat([coin_returns, market_returns, cs_mean_betas, cs_var_betas, betas, se_betas], axis=1, keys=['coin', 'market', 'cs_avg_beta', 'cs_var_beta', 'beta', 'se_beta']).dropna()

    # compute the weight that shrink the beta of the coin from cross-sectional information
    beta_prior, var_beta_prior, var_beta_ols, beta = aligned['cs_avg_beta'].values, aligned['cs_var_beta'].values, aligned['se_beta'].values ** 2, aligned['beta'].values
    ws = var_beta_prior / (var_beta_prior + var_beta_ols)
    beta_shrinckage = ws*beta + (1-ws)*beta_prior
    aligned['beta_vck'] = beta_shrinckage
    
    shared = pd.concat([coin_returns, market_returns], join='inner', axis=1, keys=['coin', 'market']).dropna()
    idx = shared.index
    n = len(idx)
    result = _get_empty_result(idx)

    start_idx, end_idx = shared.index.searchsorted(aligned.index.min()), shared.index.searchsorted(aligned.index.max()) + 1

    for i in range(start_idx, end_idx):
        if idx[i] not in aligned.index : continue
        k = i - window + 1

        y, x = shared['coin'].iloc[k:i+1].values, shared['market'].iloc[k:i+1].values
        #beta_vck_i = aligned.loc[aligned.index == idx[i],'beta_vck'].values[0]
        beta_vck_i = aligned.at[idx[i],'beta_vck']
        result_eval = _evalute_ols(y, x, beta_vck_i)
        result.loc[result.index[i]] = [result_eval['alpha'], result_eval['beta'], result_eval['r2'], result_eval['se_beta'], result_eval['res_vol']]

    return result.dropna(subset=["beta"])


def compute_rolling_realized_beta(coin_returns : pd.Series, market_returns : pd.Series, horizon : int) -> pd.Series :
    '''
    Compute the rolling realized beta between returns of the coin and market over an horizon

    Return the rolling beta
    '''

    shared = pd.concat([coin_returns, market_returns], join='inner', axis=1, keys=['coin', 'market']).dropna()
    idx = shared.index
    results = pd.Series(index=idx, dtype= float)
    n = len(idx)

    for i in range(n - horizon):
        start, end = i + 1, i+1+horizon
        y,x = shared['coin'].iloc[start:end].values, shared['market'].iloc[start:end].values
        X = sm.add_constant(x)
        res = sm.OLS(y, X).fit()
        results.iloc[i] = res.params[1]

    return results.dropna()

def compute_rolling_realized_betas_all_windows(filtering_in_index : bool, in_index : pd.DataFrame, coins_returns : pd.DataFrame, market_returns : pd.Series, horizons : list[int]) -> Dict[int, Dict[str, pd.Series]]:
    '''
    Compute rolling realized betas over each coin for a list of specified horizons (OLS)

    Returns a data structure that encapsulates for each coin a series of realized betas
    '''

    results = {}

    for horizon in horizons:
        res = compute_rolling_realized_betas(filtering_in_index, in_index, coins_returns, market_returns, horizon)
        if len(res) < 1 :
            print(f'error during compute_rolling_realized_betas for horizon [{horizon}] -> no result')
            continue
        results[horizon] = res
    
    return results


def compute_rolling_realized_betas(filtering_in_index : bool, in_index : pd.DataFrame, coins_returns : pd.DataFrame, market_returns : pd.Series, horizon : int) -> Dict[str, pd.Series]:
    '''
    Compute rolling realized betas over each coin for a specified horizon (OLS)

    Returns a data structure that encapsulates for each coin a series of realized betas
    '''
    results = {}
    coins = [coin for coin in coins_returns.columns]

    for coin in coins:
        if coin not in coins_returns: continue
        coin_returns = coins_returns[coin]
        try:
            res = compute_rolling_realized_beta(coin_returns, market_returns, horizon)
            if len(res) < 1:
                print(f'error during compute_rolling_realized_beta for coin [{coin}] -> no result')
                continue

            if filtering_in_index:
                    if coin not in in_index.columns:
                        print(f'error during computation realized beta for coin [{coin}] -> not in in_index df')
                        continue
                    mask = in_index[coin].reindex(res.index).fillna(False)
                    res = res.loc[mask]

                    if len(res) < 1:
                        print(f'error during computation realized beta for coin [{coin}] -> after mask-out according in_index df, res is empty')
                        continue

            results[coin] = res

        except Exception as ex: print(f'exception raised during compute_rolling_realized_beta for coin [{coin}]: {ex}')

    return results



def compute_rolling_betas(filtering_in_index : bool, in_index : pd.DataFrame, coins_returns : pd.DataFrame, market_returns : pd.Series, market_ohlc : pd.DataFrame, window:int, halflife: int, estimators : List['str']=None, delta : int=3) -> Dict[str, Dict[str, pd.DataFrame]] :
    '''
    Compute betas for all coins for a particular window and specified estimators
    Available estimators: ols, sw, wls, vol_range, vck

    Returns a data structure that contains stats result [alpha, beta, r², se_beta, res_vol] for each beta estimator and coin
    '''
    if estimators is None: estimators = helper.DEFAULT_ESTIMATORS

    # build a method map, for dynamically iterate each estimator
    estimator_map = {
        'ols' : lambda c, m : compute_rolling_ols_beta(c, m, window=window),
        'sw' : lambda c, m : compute_sw_rolling_beta(c, m, window=window, delta=delta),
        'wls' : lambda c, m : compute_wls_rolling_beta(c, m, window=window, halflife=halflife),
        'vr' : lambda c, m : compute_vol_range_rolling_beta(c, m, markets=market_ohlc, window=window)
    }
    
    results = { estimator : {} for estimator in estimators }
    coins = coins_returns.columns
    
    # progress
    total = len(coins) * len(estimators)
    done = 0

    # iterate each coin / estimator
    for coin in coins:

        coin_returns = coins_returns[coin].dropna()

        m = len(coin_returns)
        if m <= window:
            print(f'Impossible to compute an estimator for [{coin}] due to nb returns {m} < window {window}')
            continue
        
        for estimator_str in estimators:
            if estimator_str not in estimator_map: continue
            
            try:
                # estimate rolling beta
                res = estimator_map[estimator_str](coin_returns, market_returns)
                if len(res) < 1: 
                    print(f'error during {estimator_str} for coin [{coin}] -> no result')
                    continue
                
                if filtering_in_index:
                    if coin not in in_index.columns:
                        print(f'error during {estimator_str} for coin [{coin}] -> not in in_index df')
                        continue
                    mask = in_index[coin].reindex(res.index).fillna(False)
                    res = res.loc[mask]

                    if len(res) < 1:
                        print(f'error during {estimator_str} for coin [{coin}] -> after mask-out according in_index df, res is empty')
                        continue

                results[estimator_str][coin] = res
                

            except Exception as ex: print(f'error occured {estimator_str} for coin [{coin}]: {ex}')
            finally:
                done += 1
                if done % 20 == 0: print(f'{done}/{total} {done*100.0/total}%')

    if 'vck' not in estimators :
        return results

    # special case for vck method that required the computation of ols beta of all coin
    if 'vck' in estimators and 'ols' not in estimators:
        print('impossible to compute the vck estimator if ols estimator is not computed before! Include ols!')
        return results
    
    beta_df = pd.DataFrame(index=coins_returns.index, columns=coins, dtype=float)
    ols_coins = results['ols'].keys()
    for coin in ols_coins:
        if coin not in results['ols']:
            print(f'skip {coin} for cross-sectional computation due to {coin} des not match into ols results')
            continue
        range_beta = results['ols'][coin].index
        beta_df.loc[range_beta ,coin] = results['ols'][coin]['beta'].values
    beta_df['beta_mean'] = beta_df.mean(axis=1) # compute cross-sectional beta ols mean
    beta_df['beta_std'] = beta_df.std(axis=1) # compute cross-sectional beta ols std
    cs_mean, cs_vol = beta_df['beta_mean'].dropna(), beta_df['beta_std'].dropna()

    total = len(coins)
    done = 0

    for coin in ols_coins:

        if coin not in results['ols']:
            print(f'skip {coin} for cross-sectional computation due to {coin} des not match into ols results')
            continue

        coin_returns = coins_returns[coin].dropna()

        m = len(coin_returns)
        if m <= window:
            print(f'Impossible to compute an estimator for [{coin}] due to nb returns {m} < window {window}')
            continue
        
        result_coin_ols = results['ols'][coin]
        try:
            res = compute_vasicek_shrinkage_rolling_beta(coin_returns, market_returns, cs_mean, cs_vol, result_coin_ols['beta'], result_coin_ols['se_beta'], window=window)
            if len(res) < 1: 
                print(f'error during vck for coin [{coin}] -> no result')
                continue

            if filtering_in_index:
                    if coin not in in_index.columns:
                        print(f'error during vck for coin [{coin}] -> not in in_index df')
                        continue
                    mask = in_index[coin].reindex(res.index).fillna(False)
                    res = res.loc[mask]

                    if len(res) < 1:
                        print(f'error during vck for coin [{coin}] -> after mask-out according in_index df, res is empty')
                        continue

            results['vck'][coin] = res

        except Exception as ex: print(f'error occured vck for coin [{coin}]: {ex}')
        finally: 
            done += 1
            if done % 20 == 0: print(f'{done}/{total} {done*100.0/total}%')
            
    return results

def compute_multi_windows_beta(
        filtering_in_index : bool,
        in_index : pd.DataFrame, 
        coins_returns : pd.DataFrame, 
        market_returns : pd.Series,
        market_ohlc : pd.DataFrame, 
        windows: list[int],
        estimators : List['str']=None, 
        delta : int=3) -> Dict[int, Dict[str, Dict[str, pd.DataFrame]]] :

    '''
    Compute for each window, several estimators for each coins

    Returns a data structure that contains stats result [alpha, beta, r², se_beta, res_vol] for each window, beta estimator and coin
    '''
    results = {}

    for window in windows:
        halflife = window//2
        res = compute_rolling_betas(filtering_in_index=filtering_in_index, in_index=in_index, coins_returns=coins_returns, market_returns=market_returns, market_ohlc=market_ohlc, window=window, halflife=halflife, estimators=estimators, delta=delta)
        if len(res) < 1: continue

        results[window] = res
    
    return results

if __name__ == "__main__":
    data = data_handler.load_daily_data()

    coins_returns = data["coins_returns"]
    market_returns = data["market_returns"]
    market_ohlc = data["market_ohlc"]

    # lead some beta estimation
    window = 252
    res_ols = compute_rolling_ols_beta(coins_returns['bitcoin'], market_returns, window)
    print(res_ols.tail(10))
    res_sw = compute_sw_rolling_beta(coins_returns['bitcoin'], market_returns, window)
    print(res_sw.tail(10))
    res_wls = compute_wls_rolling_beta(coins_returns['bitcoin'], market_returns, window, halflife=window//2)
    print(res_wls.tail(10))
    res_vr = compute_vol_range_rolling_beta(coins_returns['bitcoin'], market_returns, market_ohlc, window)
    print(res_vr.tail(10))
    print('** END TEST **')
    
