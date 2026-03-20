from enum import Enum

# **************** FREQUENCY ****************
class ESTIMATION_FREQ(Enum):
    DAILY = 1
    WEEKLY = 2

# **************** DATA ****************
DAILY_DATA_FILE_PATH = ''
WEEKLY_DATA_FILE_PATH = ''


# *************** COMPUTATION *************
ZERO_APPROX = 1e-15


# *************** ESTIMATOR ***************
DEFAULT_ESTIMATORS = ['ols', 'sw', 'wls', 'vol_range', 'vck']