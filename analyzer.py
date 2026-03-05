import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from scipy import stats as sp_stats
import statsmodels.api as sm
import time
from datetime import datetime, timedelta, timezone

import helper
import data_handler
import beta_estimators
import portfolio_builder


def eom_resampling(coins_returns : pd.DataFrame, market_returns : pd.Series, daily_estimated_betas : Dict[str, Dict[str, pd.DataFrame]], daily_realized_betas : Dict[str, pd.Series], frequency : helper.ESTIMATION_FREQ) -> Tuple[Dict[str, Dict[str, pd.DataFrame]], Dict[str, pd.Series]] :
    '''
    Resample to each end of month (follow Sila 2025) the daily/weekly estimated betas & daily realized betas dataset

    Returns a tuple that contains both resampled dataset
    '''
    aligned_idx = coins_returns.index.intersection(market_returns.index)
    eoms = pd.DatetimeIndex([ aligned_idx[aligned_idx.to_period('M') == m][-1] for m in aligned_idx.to_period('M').unique()])

    '''
    if frequency == helper.ESTIMATION_FREQ.DAILY: eoms = pd.DatetimeIndex([ aligned_idx[aligned_idx.to_period('M') == m][-1] for m in aligned_idx.to_period('M').unique()])
    elif frequency == helper.ESTIMATION_FREQ.WEEKLY: eoms = pd.DatetimeIndex([aligned_idx[aligned_idx.to_period('M') == m][-1] for m in aligned_idx.to_period('M').unique()]) # same logic — last available date per calendar month
    '''

    eom_estimated_betas, eom_realized_betas = {}, {}

    # build the eom estimated betas &
    for window, daily_estimators_estimated_betas in daily_estimated_betas.items():

        if window not in daily_realized_betas:
            print(f'w={window} is present in estimated_betas but not in realized_betas')
            continue
        eom_estimated_betas[window] = {}
        for estimator, daily_estimated_coins_betas in daily_estimators_estimated_betas.items():

            eom_estimated_betas[window][estimator] = {}
            for coin, daily_coin_estimated_results in daily_estimated_coins_betas.items():
                valid_eoms = eoms.intersection(daily_coin_estimated_results.index)
                eom_estimated_betas[window][estimator][coin] = daily_coin_estimated_results.loc[valid_eoms]

    # build  eom realized betas
    for window, daily_coins_realized_betas in daily_realized_betas.items():
        eom_realized_betas[window] = {}
        for coin, daily_coin_realized in daily_coins_realized_betas.items():
            valid_eoms = eoms.intersection(daily_coin_realized.index)
            eom_realized_betas[window][coin] = daily_coin_realized.loc[valid_eoms]

    return eom_estimated_betas, eom_realized_betas

def compute_cross_sectional_stats(betas_panel: pd.DataFrame) -> pd.DataFrame:
    """
    Compute cross-sectional statistics of betas over time.
    
    For each date: mean, median, std, min, max of betas across all coins.
    
    Returns DataFrame with one row per date.
    """
    stats = pd.DataFrame(index=betas_panel.index)
    stats["mean"] = betas_panel.mean(axis=1)
    stats["median"] = betas_panel.median(axis=1)
    stats["std"] = betas_panel.std(axis=1)
    stats["min"] = betas_panel.min(axis=1)
    stats["max"] = betas_panel.max(axis=1)
    stats["q25"] = betas_panel.quantile(0.25, axis=1)
    stats["q75"] = betas_panel.quantile(0.75, axis=1)
    stats["n_coins"] = betas_panel.count(axis=1)

    return stats.dropna(subset=["mean"])

def get_column_from_panel(estimator_results: Dict[str, pd.DataFrame], col: str = "beta") -> pd.DataFrame:
    """
    Convert {coin: DataFrame} to a {coin : series}
    columns = coins, index = dates, values = defined column.
    """
    series_dict = {}
    for coin, df in estimator_results.items():
        if col in df.columns:
            series_dict[coin] = df[col]

    if not series_dict: return pd.DataFrame()

    return pd.DataFrame(series_dict)

def compute_pooled_panel_regression(estimated_betas : Dict[int, Dict[str, Dict[str, pd.DataFrame]]], realized_betas : Dict[int, Dict[str, pd.Series]]) -> pd.DataFrame :
    '''
    Compute the pooled panel regression per window and estimator. The panel is composed of all coins and dates

    Returns a dataframe with statistics per window and estimator
    '''
    results = []

    for window, estimators_estimated_betas in estimated_betas.items():

        if window not in realized_betas:
            print(f'w={window} is present in estimated_betas but not in realized_betas, no added to the pooled panel')
            continue
        
        coins_realized_betas = realized_betas[window]

        for estimator, coins_estimated_betas in estimators_estimated_betas.items():

            all_x, all_y, all_se = [], [], []

            for coin, coin_estimated_betas in coins_estimated_betas.items():

                if coin not in coins_realized_betas:
                    print(f'w={window} estimator: {estimator}: {coin} is present in estimated_betas but not in realized_betas, no added to the pooled panel')
                    continue

                coin_realized_beta = coins_realized_betas[coin]
                common = coin_estimated_betas.index.intersection(coin_realized_beta.index)

                if len(common) < 1: 
                    print(f'w={window}  estimator: {estimator}: {coin} does not have common index date (estimated w realized)')
                    continue

                estimated = coin_estimated_betas.loc[common].dropna()
                realized = coin_realized_beta.loc[common].dropna()
                
                common = estimated.index.intersection(realized.index)
                if len(common) < 1: 
                    print(f'w={window} estimator: {estimator}: {coin} does not have common index date (estimated w realized)')
                    continue

                all_x.append(estimated.loc[common, 'beta'].values)
                all_y.append(realized.loc[common].values)

                # SE for CEV bias
                all_se.append(estimated.loc[common, 'se_beta'].values)

            if len(all_x) < 1:
                print(f'w={window} estimator: {estimator}: none elt to regress!')
                continue

            # Pooled OLS with Newey-West HAC SE
            x, y = np.concatenate(all_x), np.concatenate(all_y)

            n = len(x)
            print(f'w={window} {estimator}: lead pooled panel regression for {n} rows')

            X = sm.add_constant(x)
            model = sm.OLS(y, X)
            # NW lags: ceil(w/22) - 1, minimum 0
            nw_lags = max(0, int(np.ceil(window / 22)) - 1)
            fit_nw = model.fit(cov_type="HAC", cov_kwds={"maxlags": nw_lags})

            gamma_0 = fit_nw.params[0]
            gamma_beta = fit_nw.params[1]
            se_gamma_nw = fit_nw.bse[1]
            r2 = fit_nw.rsquared

            # Also naive OLS SE for comparison
            fit_ols = model.fit()
            se_gamma_ols = fit_ols.bse[1]

            # Error metrics (raw predictor as forecast)
            raw_errors = y - x
            rmse = np.sqrt(np.mean(raw_errors ** 2))
            mae = np.mean(np.abs(raw_errors))
            abs_y = np.abs(y)
            mape = np.mean(np.abs(raw_errors) / np.maximum(abs_y, 1e-6))
            smape = np.mean(2 * np.abs(raw_errors) / np.maximum(np.abs(x) + abs_y, 1e-6))

            # CEV bias
            cev_bias = np.nan
            if all_se:
                se_all = np.concatenate(all_se)
                sigma2_x = np.var(x, ddof=1)
                sigma2_se = np.mean(se_all ** 2)
                if sigma2_x + sigma2_se > 1e-15: cev_bias = 1 - sigma2_x / (sigma2_x + sigma2_se)

            results.append({
                "window": window,
                "estimator": estimator,
                "gamma_0": gamma_0,
                "gamma_beta": gamma_beta,
                "se_gamma_nw": se_gamma_nw,
                "se_gamma_ols": se_gamma_ols,
                "nw_lags": nw_lags,
                "r_squared": r2,
                "rmse": rmse,
                "mae": mae,
                "mape": mape,
                "smape": smape,
                "cev_bias": cev_bias,
                "n_obs": n,
            })


    return pd.DataFrame() if len(results) < 1 else pd.DataFrame(results).set_index(["window", "estimator"]).sort_index()

def analyze_beta_dispersion(estimators : list[str], estimator_betas : Dict[int, Dict[str, Dict[str, pd.DataFrame]]], market_returns : pd.Series, rolling_window : int = 60) -> pd.DataFrame :
    '''
    Analyze the relationship between the beta dispersion and the market volatility & beta dispersion according market regime

    Compute the correlation between beta dispersion and rolling market volatility over a period (vol_window as parameter, 60 days)
    Build two masks for classifying high and low vol regime such that:
    - high vol if market vol > top 25% of market_vol
    - low vol if market vol < bottom 25% of market_vol
    Define bull/bear by cumulative market return over trailing window
    - bull if mkt cumulative return > 0
    - otherwise bear
    Compute beta mean and dispersion according the regime identified

    Returns a dataframe: window | estimator | vol_beta_high_vol | vol_beta_low_vol | corr_vol_beta_vol_market | mean_beta_bull | mean_beta_bear | vol_beta_bull | vol_beta_bear
    '''
    #results = pd.DataFrame(columns=['window', 'estimator', 'vol_beta_high_vol', 'vol_beta_low_vol', 'corr_vol_beta_vol_market'], dtype=float)
    results = []
    # compute market rolling volatility & rolling market cumulative
    mkt_vol = market_returns.rolling(rolling_window).std() * np.sqrt(365)
    mkt_cumret = (1 + market_returns).rolling(60).apply(lambda x: x.prod() - 1)
    bull = mkt_cumret > 0
    #bear = mkt_cumret <= 0

    for window in estimator_betas.keys():
        for estimator in estimators:
            estimated_betas = get_column_from_panel(estimator_betas[window][estimator], col = 'beta')

            # Align both series on common dates
            cs_std = estimated_betas.std(axis=1)
            cs_mean = estimated_betas.mean(axis=1)
            
            aligned = pd.concat([cs_std.rename('cs_std'), cs_mean.rename('cs_mean'), mkt_vol.rename('mkt_vol'), bull.rename('bull')], axis=1).dropna()

            # volatility part
            corr = aligned['cs_std'].corr(aligned['mkt_vol'])
            high_vol = aligned['mkt_vol'] > aligned['mkt_vol'].quantile(0.75)
            low_vol = aligned['mkt_vol'] < aligned['mkt_vol'].quantile(0.25)
            mean_high_vol, mean_low_vol = aligned.loc[high_vol, 'cs_std'].mean(), aligned.loc[low_vol, 'cs_std'].mean()

            # regime part
            mean_beta_bull, mean_beta_bear = aligned.loc[aligned['bull'], 'cs_mean'].mean(), aligned.loc[~aligned['bull'], 'cs_mean'].mean()
            vol_beta_bull, vol_beta_bear = aligned.loc[aligned['bull'], 'cs_std'].mean(), aligned.loc[~aligned['bull'], 'cs_std'].mean()

            results.append({
                'window' : window,
                'estimator' : estimator,
                'vol_beta_high_vol' : mean_high_vol,
                'vol_beta_low_vol' : mean_low_vol,
                'corr_vol_beta_vol_market' : corr,
                'mean_beta_bull' : mean_beta_bull,
                'mean_beta_bear' : mean_beta_bear,
                'vol_beta_bull' : vol_beta_bull,
                'vol_beta_bear' : vol_beta_bear
            })

            #print(f"[w= {window};mthd= {estimator}] Correlation (beta dispersion vs market vol): {corr:.3f}")
            #print(f"[w= {window};mthd= {estimator}] Beta dispersion (high vol): {aligned.loc[high_vol, 'cs_std'].mean():.3f}")
            #print(f"[w= {window};mthd= {estimator}] Beta dispersion (low vol):  {aligned.loc[low_vol, 'cs_std'].mean():.3f}")

    return pd.DataFrame() if len(results) < 1 else pd.DataFrame(results).set_index(["window", "estimator"]).sort_index()

def build_transition_matrix_all(estimators_betas : Dict[int, Dict[str, Dict[str, pd.DataFrame]]], n_quantiles : int = 5, period : str = 'M') -> Tuple[Dict[int, Dict[str, np.array]], Dict[int, Dict[str, np.array]], Dict[int, Dict[str, Any]], pd.DataFrame] :
    '''
    Build the transition matrix for each couple window/estimator, following the methodology from "Beta stability and portfolio formation" (Levy 1994 / Brooks, Faff & Lee 1998).
    Returns per window and estimator:
    - transition matrix
    - transition prob. matrix
    - statistics
    - data structure that synthetize for stats for all couples
    '''
    results = []
    trans_matrices, trans_prob_matrices, all_stats = {}, {}, {}

    for window, estimators_window_betas in estimators_betas.items() :
        trans_matrices[window] = {}
        trans_prob_matrices[window] = {}
        all_stats[window] = {}

        for estimator, estimator_coins_betas in estimators_window_betas.items():
            trans_matrix, trans_probs_matrix, stats = build_transition_matrix(estimator_coins_betas, n_quantiles, period)
            trans_matrices[window][estimator] = trans_matrix
            trans_prob_matrices[window][estimator] = trans_probs_matrix
            all_stats[window][estimator] = stats
            results.append({
                'window': window,
                'estimator' : estimator,
                'diagonal_mean': stats['diagonal_mean'],
                'off_diagonal_mean': stats['off_diagonal_mean'],
                'extreme_stickiness': stats['extreme_stickiness'],
                'interpretation' : stats['interpretation']
            })

    return trans_matrices, trans_prob_matrices, all_stats, pd.DataFrame() if len(results) < 1 else pd.DataFrame(results).set_index(["window", "estimator"]).sort_index()

def build_transition_matrix(estimated_coins : Dict[str, pd.DataFrame], n_quantiles : int = 5, period : str = 'M') -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]: 
    '''
    Following the methodology from "Beta stability and portfolio formation" (Levy 1994 / Brooks, Faff & Lee 1998):

    1. At each month-end, rank all CCI30 coins by beta -> assign to quintiles (Q1 = lowest, Q5 = highest)
    2. At next month-end, re-rank -> new quintiles
    3. Count transitions: how many coins moved from quintile i to quintile j
    4. Normalize rows to probabilities

    - Strong diagonal (> 0.35): STABLE — past beta predicts future beta
    - Weak diagonal (~0.20): UNSTABLE — betas are random walks
    - Corner stickiness (Q1, Q5 > Q3): extreme betas are more persistent

    Returns:
    - transition matrix (count), transition probabilities, dictionary of stats:
    
    - diagonal_mean: average probability of staying in same quintile
    - diagonal_values: per-quintile stickiness
    - off_diagonal_mean: average probability of moving
    - extreme_stickiness: average of Q1 and Q5 diagonal (corners)
    - middle_stickiness: Q3 diagonal
    '''

    # get beta
    estimated_coins_betas = get_column_from_panel(estimated_coins, 'beta')
    # Resample to period-end betas
    periodic = estimated_coins_betas.resample(period).last()
    # Drop periods with too few coins
    min_coins = n_quantiles * 2  # need at least 2 per bin
    periodic = periodic.dropna(axis=0, thresh=min_coins)

    # Assign quintiles at each period
    def assign_quantiles(row):
        valid = row.dropna()
        if len(valid) < n_quantiles: return pd.Series(dtype=float)
        try:
            return pd.qcut(valid.rank(method="first"), n_quantiles, labels=False)
        except ValueError:
            # Handle ties: use rank-based assignment
            ranks = valid.rank(method="first")
            return pd.cut(ranks, n_quantiles, labels=False)
        
    quintiles = periodic.apply(assign_quantiles, axis=1)

    # Build transition counts
    transitions = np.zeros((n_quantiles, n_quantiles))

    for t in range(len(quintiles) - 1):
        current = quintiles.iloc[t].dropna()
        next_q = quintiles.iloc[t + 1].dropna()
        common = current.index.intersection(next_q.index)

        for coin in common:
            i = int(current[coin])
            j = int(next_q[coin])
            if 0 <= i < n_quantiles and 0 <= j < n_quantiles:
                transitions[i, j] += 1

    # Normalize rows
    row_sums = transitions.sum(axis=1, keepdims=True)
    transition_probs = np.divide(transitions, row_sums, out=np.zeros_like(transitions), where=row_sums > 0)

    # compute statistics
    n = transition_probs.shape[0]
    diag = np.diag(transition_probs)
    stats = {
        "diagonal_mean": np.mean(diag),
        "diagonal_values": diag.tolist(),
        "off_diagonal_mean": (transition_probs.sum() - np.trace(transition_probs)) / (n * n - n),
        "extreme_stickiness": (diag[0] + diag[-1]) / 2,
        "middle_stickiness": diag[n // 2],
        "interpretation": (
            "STABLE" if np.mean(diag) > 0.35 
            else "MODERATE" if np.mean(diag) > 0.25 
            else "UNSTABLE"
        )
    }
    return transitions, transition_probs, stats

if __name__ == "__main__":
    # TEST
    start_recorder = time.perf_counter()
    data = data_handler.load_daily_data()

    #sample_coins_returns = data['coins_returns'].loc[:,['aave', 'bitcoin', 'ethereum', 'xrp', 'solana']]
    #sample_coins_returns = data['coins_returns'].loc[:,['aave']]
    sample_coins_returns = data['coins_returns']
    #sample_coins_returns = data['coins_returns'].loc[:, data['coins_returns'].columns[:20]]
    

    # run a test with ols
    estimators=['ols', 'sw', 'wls', 'vr', 'vck']
    #estimators = ['ols', 'wls']
    windows = [365, 180, 90, 30]
    #windows = [365, 30]
    #windows = [365]
    window = 365
    #daily_estimators_betas = beta_estimators.compute_rolling_betas(True, data['in_index'], sample_coins_returns, data['market_returns'], data['market_ohlc'], window=window, halflife=window//2, estimators=['ols', 'sw', 'wls', 'vr', 'vck']) #estimators=['ols', 'sw', 'wls', 'vr', 'vck']
    #daily_realized_betas = beta_estimators.compute_rolling_realized_betas(True, data['in_index'], sample_coins_returns, data['market_returns'], horizon=window)

    #daily_estimators_betas = beta_estimators.compute_multi_windows_beta(True, data['in_index'], sample_coins_returns, data['market_returns'], data['market_ohlc'], windows, estimators=['ols', 'sw', 'wls', 'vr', 'vck'])
    daily_estimators_betas = beta_estimators.compute_multi_windows_beta(True, data['in_index'], sample_coins_returns, data['market_returns'], data['market_ohlc'], windows, estimators=estimators) #estimators=['ols', 'sw', 'wls', 'vr', 'vck']
    daily_realized_betas = beta_estimators.compute_rolling_realized_betas_all_windows(True, data['in_index'], sample_coins_returns, data['market_returns'], windows)

    sample_coins_returns_masked = data_handler.mask_out_prices_no_constituents(sample_coins_returns, data['in_index'])
    eom_estimator_betas, eom_realized_betas = eom_resampling(sample_coins_returns_masked, data['market_returns'], daily_estimators_betas, daily_realized_betas, frequency=helper.ESTIMATION_FREQ.DAILY)
    if len(eom_estimator_betas) < 1: print('error: eom_estimator_betas is empty')
    if len(eom_realized_betas) < 1: print('error: eom_realized_betas is empty')

    #eom_window_estimator_betas = {window : eom_estimator_betas}
    #eom_window_realized_betas = {window : eom_realized_betas}

    res_pooled_panel_regression = compute_pooled_panel_regression(eom_estimator_betas, eom_realized_betas)
    if res_pooled_panel_regression.empty: print('error during compute_pooled_panel_regression')
    else: print(res_pooled_panel_regression)
    
    res_pooled_panel_regression.to_csv(f'pooled_panel_regression - results - {datetime.now().strftime('%Y-%m-%d %H%M%S')}.csv', sep=';')

    # compute stats about avg beta for each window / estimator
    #estimator_methds = res_pooled_panel_regression.index.get_level_values('estimator').unique()
    res_dispersion_beta = analyze_beta_dispersion(estimators, daily_estimators_betas, data['market_returns'], rolling_window=60)
    if res_dispersion_beta.empty: print('error during analyze_beta_dispersion')
    else: print(res_dispersion_beta)
    res_dispersion_beta.to_csv(f'analyze_beta_dispersion - results - {datetime.now().strftime('%Y-%m-%d %H%M%S')}.csv', sep=';')

    # compute matrix transition & statistics
    trans_matrices, trans_prob_matrices, all_stats, res_transition_df = build_transition_matrix_all(daily_estimators_betas, n_quantiles=5, period='M')
    if res_transition_df.empty: print('error during build_transition_matrix_all')
    else: print(res_transition_df.round(2))
    
    res_transition_df.to_csv(f'transiton matrix - results - {datetime.now().strftime('%Y-%m-%d %H%M%S')}.csv', sep=';')

    # ************************************ build ew portfolio ************************************************
    ew_returns = portfolio_builder.build_ew_portfolio_returns(sample_coins_returns, data['in_index'])
    ew_returns_df = ew_returns.to_frame()

    daily_ew_estimators_betas = beta_estimators.compute_multi_windows_beta(False, data['in_index'], ew_returns_df, data['market_returns'], data['market_ohlc'], windows, estimators=estimators) #estimators=['ols', 'sw', 'wls', 'vr', 'vck']
    daily_ew_realized_betas = beta_estimators.compute_rolling_realized_betas_all_windows(False, data['in_index'], ew_returns_df, data['market_returns'], windows)

    eom_ew_estimator_betas, eom_ew_realized_betas = eom_resampling(ew_returns_df, data['market_returns'], daily_ew_estimators_betas, daily_ew_realized_betas, frequency=helper.ESTIMATION_FREQ.DAILY)
    if len(eom_ew_estimator_betas) < 1: print('error: eom_ew_estimator_betas is empty')
    if len(eom_ew_realized_betas) < 1: print('error: eom_ew_realized_betas is empty')

    # save dataframe
    for window in windows:
        for estimator in estimators:
            eom_ew_estimator_betas[window][estimator]['ew_portfolio'].to_csv(f'EW - {window} {estimator} estimated {datetime.now().strftime('%Y-%m-%d %H%M%S')}.csv', sep=";")
        eom_ew_realized_betas[window]['ew_portfolio'].to_csv(f'EW - {window} realized {datetime.now().strftime('%Y-%m-%d %H%M%S')}.csv', sep=";")

    res_pooled_panel_regression = compute_pooled_panel_regression(eom_ew_estimator_betas, eom_ew_realized_betas)
    if res_pooled_panel_regression.empty: print('error during compute_pooled_panel_regression')
    else: print(res_pooled_panel_regression)
    
    res_pooled_panel_regression.to_csv(f'EW - pooled_panel_regression - results - {datetime.now().strftime('%Y-%m-%d %H%M%S')}.csv', sep=';')

    # compute stats about avg beta for each window / estimator
    #estimator_methds = res_pooled_panel_regression.index.get_level_values('estimator').unique()
    res_dispersion_beta = analyze_beta_dispersion(estimators, daily_ew_estimators_betas, data['market_returns'], rolling_window=60)
    if res_dispersion_beta.empty: print('error during analyze_beta_dispersion')
    else: print(res_dispersion_beta)
    res_dispersion_beta.to_csv(f'EW - analyze_beta_dispersion - results - {datetime.now().strftime('%Y-%m-%d %H%M%S')}.csv', sep=';')

    # compute matrix transition & statistics
    trans_matrices, trans_prob_matrices, all_stats, res_transition_df = build_transition_matrix_all(daily_ew_estimators_betas, n_quantiles=5, period='M')
    if res_transition_df.empty: print('error during build_transition_matrix_all')
    else: print(res_transition_df.round(2))
    
    res_transition_df.to_csv(f'EW - transiton matrix - results - {datetime.now().strftime('%Y-%m-%d %H%M%S')}.csv', sep=';')

    print(f'Elapsed time: {time.perf_counter() - start_recorder:.6f} seconds')

    print('*********** END OF TEST ***********')
