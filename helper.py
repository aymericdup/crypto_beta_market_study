from enum import Enum

# **************** FREQUENCY ****************
class ESTIMATION_FREQ(Enum):
    DAILY = 1
    WEEKLY = 2

# **************** DATA ****************
DAILY_DATA_FILE_PATH = 'D:/Personal/Jobs/prepa interviews/Crypto united/cci30_daily.csv'
WEEKLY_DATA_FILE_PATH = 'D:/Personal/Jobs/prepa interviews/Crypto united/cci30_weekly.csv'


# *************** COMPUTATION *************
ZERO_APPROX = 1e-15


# *************** ESTIMATOR ***************
DEFAULT_ESTIMATORS = ['ols', 'sw', 'wls', 'vol_range', 'vck']