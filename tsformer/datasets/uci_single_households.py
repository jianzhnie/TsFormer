import io
import logging
import os
import warnings
import zipfile
from datetime import datetime

import numpy as np
import pandas as pd
import requests
from sklearn.preprocessing import StandardScaler

from tsformer.utils.utils import set_datetime_index

logger = logging.getLogger('log')

NAME = 'uci'
SAMPLES_PER_DAY = 96
FREQ = '15T'
TARGET = 'Global_active_power'
DATETIME = 'datetime'
config = {'data': 'uci'}


def download():
    """
    Download data and extract them in the data directory
    :return:
    """
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip'
    logger.info('Donwloading .zip file from {}'.format(url))
    r = requests.get(url)
    logger.info('File downloaded.')
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(path=config['data'])
    new_loc = os.path.join(config['data'],
                           'UCI_household_power_consumption.csv')
    os.rename(
        os.path.join(config['data'], 'household_power_consumption.txt'),
        new_loc)
    logger.info('File extracted and available at {}'.format(new_loc))
    process_csv()
    logger.info(
        'The original CSV has been parsed and is now available at {}'.format(
            os.path.join(config['data'],
                         'UCI_household_power_consumption_synth.csv')))


def process_csv():
    """Parse the datetime field, Sort the values accordingly and save the new
    dataframe to disk."""
    df = pd.read_csv(
        os.path.join(config['data'], 'UCI_household_power_consumption.csv'),
        sep=';')
    df[DATETIME] = list(
        map(
            lambda d: datetime.combine(
                datetime.strptime(d[0], '%d/%m/%Y').date(),
                datetime.strptime(d[1], '%H:%M:%S').time()),
            df[['Date', 'Time']].values))
    df = df.sort_values([DATETIME]).reset_index(drop=True)
    df = df[[DATETIME, TARGET]]
    df[DATETIME] = pd.to_datetime(df[DATETIME], utc=False)

    def parse(x):
        try:
            return np.float64(x)
        except ValueError:
            return np.nan

    df[TARGET] = df[TARGET].apply(lambda x: parse(x))
    df.to_csv(
        os.path.join(config['data'],
                     'UCI_household_power_consumption_synth.csv'),
        index=False)


def load_raw_dataset():
    """
    Load the dataset as is
    :return: pandas.DataFrame: sorted dataframe with parsed datetime
    """
    df = pd.read_csv(
        os.path.join(config['data'], 'UCI_household_power_consumption.csv'),
        sep=';')
    return df


def load_dataset(fill_nan='median', get_dates_dict=False):
    """Load the dataset resampled at a frequency of 15 minutes.

    :param fill_nan: string that identifies how NaN values should be filled. Options are:
        -bfill: fill NaN value at index i with value at index i-1
        -ffill: fill NaN value at index i with value at index i+1
        -mean: fill NaN value at index i  with the mean value over all dataset at the same hour,minute
        -median: fill NaN value at index i  with the median value over all dataset at the same hour,minute
        -drop: drop all rows with missing values
    :param get_dates_dict: if True, return also the dict containing the map
                           between each dataframe index to a datetime
    :return:
        - pandas.DataFrame: sorted dataframe with parsed datetime
        - dict: map each dataframe index to a datetime
    """
    df = pd.read_csv(
        os.path.join(config['data'],
                     'UCI_household_power_consumption_synth.csv'))
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values([DATETIME]).reset_index(drop=True)
    df = df[[DATETIME, TARGET]]
    df[DATETIME] = pd.to_datetime(df[DATETIME], utc=False)
    if fill_nan:
        df = impute_missing(
            df, method=fill_nan, values_col=TARGET, datetime_col=DATETIME)
    df = hourly_aggregate(df, freq=FREQ, datetime_col=DATETIME)
    if get_dates_dict:
        idx2date = {i: df.iloc[i][DATETIME] for i in range(df.shape[0])}
        return df, idx2date
    else:
        return df


def impute_missing(df,
                   method='bfill',
                   values_col='P_plus',
                   datetime_col=DATETIME):
    """Fill missing values in the dataframe.

    :param df: the dataframe
    :param method: string that identifies how NaN values should be filled. Options are:
        -bfill: fill NaN value at index i with value at index i-1
        -ffill: fill NaN value at index i with value at index i+1
        -mean: fill NaN value at index i  with the mean value over all dataset at the same hour,minute
        -median: fill NaN value at index i  with the median value over all dataset at the same hour,minute
        -drop: drop all rows with missing values
    :param values_col: string that identfies the taregt column of the datetime in df (the quantity of interest)
    :param datetime_col: string that identifies the column of datetime in df
    :return: pandas.DataFrame with the filled values
    """

    def _group_values(df, datetime_col, values_col, by='D'):
        if by == 'D':
            df_copy = set_datetime_index(df, datetime_col).copy()
            df_copy = df_copy.dropna()
            df_copy = df_copy.groupby(
                [df_copy.index.hour,
                 df_copy.index.minute])[values_col].describe()
        else:
            raise ValueError
        return df_copy

    if method == 'bfill':
        df = df.fillna(method='bfill')
    elif method == 'ffill':
        df = df.fillna(method='ffill')
    elif method == 'mean':
        # hourly mean w/o missing values
        df_copy = _group_values(df, datetime_col, values_col, by='D')
        df[values_col] = list(
            map(
                lambda row: df_copy.loc[row[0].hour].loc[row[0].minute]['mean']
                if np.isnan(row[1]) else row[1], df[[datetime_col,
                                                     values_col]].values))
    elif method == 'median':
        # hourly median w/o missing values
        df_copy = _group_values(df, datetime_col, values_col, by='D')
        df[values_col] = list(
            map(
                lambda row: df_copy.loc[row[0].hour].loc[row[0].minute]['50%']
                if np.isnan(row[1]) else row[1], df[[datetime_col,
                                                     values_col]].values))
    elif method == 'minute_distribution':
        raise NotImplementedError()
    elif method == 'drop':
        df = df.dropna()
    else:
        raise ValueError(
            '{0}s is not a valid imputation method.'.format(method))
    return df


def hourly_aggregate(df, freq='15T', datetime_col=DATETIME):
    """Resample the dataset using a different frequency.

    :param df: the dataframe
    :param freq: string that identifies the new frequency (e.g 15T is 15 minutes, 1H is one hour)
    :param datetime_col: string that identifies the column of datetime in df
    :return: the resampled dataframe
    """
    df = set_datetime_index(df, datetime_col)
    df = df.resample(freq).mean()
    df[datetime_col] = df.index
    df = df.set_index(np.arange(df.shape[0]))
    return df


def add_exogenous_variables(df, one_hot=True):
    """Augument the dataframe with exogenous features (date/time feature +
    holiday feature). The feature's values can be kept as they are or they can
    be one hot encoded.

    :param df: the dataframe
    :param one_hot: if True, one hot encode all the features.
    :return: the matrix of exogenous features
    """
    df['year'] = df.datetime.map(lambda x: x.year)
    df['month'] = df.datetime.map(lambda x: x.month)
    df['day'] = df.datetime.map(lambda x: x.day)
    df['hour'] = df.datetime.map(lambda x: x.hour)
    df['minute'] = df.datetime.map(lambda x: x.minute)
    df['holiday'] = [0] * len(df)
    df = _add_holidays(df)
    if one_hot:
        ex_feat = pd.get_dummies(
            df, columns=['year', 'month', 'day', 'hour', 'holiday'])
        return ex_feat.values[:, -4 - (ex_feat.shape[1] - df.shape[1]):][1:]
    else:
        return df.values


def _add_holidays(df):
    """Add a binary variable to the dataset that takes value: 1 if the day is a
    holiday, 0 otherwise. Main holidays for France are considered.

    :param df: the datafrme
    :return: the agumented dtaframe
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        idx = []
        idx.extend(
            df[df.day == 1][df.month == 1].index.tolist())  # new year's eve
        idx.extend(
            df[df.day == 8][df.month == 5].index.tolist())  # ww1 victory's day
        idx.extend(
            df[df.day == 30][df.month == 5].index.tolist())  # ascension day
        idx.extend(
            df[df.day == 14][df.month == 7].index.tolist())  # bastille day
        idx.extend(df[df.day == 15][df.month ==
                                    8].index.tolist())  # assumption of mary
        idx.extend(
            df[df.day == 1][df.month == 11].index.tolist())  # all saints
        idx.extend(
            df[df.day == 11][df.month == 11].index.tolist())  # armsistice day
        idx.extend(
            df[df.day == 25][df.month == 12].index.tolist())  # christams
        df.loc[idx, 'holiday'] = 1
        return df


def transform(X, scaler=None):
    """Apply standard scaling to the input variables.

    :param X:
    :param scaler: the scaler to use, None if StandardScaler has to be used
    :return:
    """
    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(X)
    return scaler, scaler.transform(X)


def inverse_transform(X, scaler, trend=None):
    X = X.astype(np.float32)
    X = scaler.inverse_transform(X)
    try:
        X += trend
    except TypeError as e:
        logger.warn(str(e))
    except Exception as e:
        logger.warn(
            'General error (not a TypeError) while adding back time series trend. \n {}'
            .format(str(e)))
    return X


def apply_detrend(df, train_len):
    """Perform detrending on a time series by subtrating from each value of the
    dataset the average value computed over the training dataset for each
    hour/minute/weekdays.

    :param df: the dataset
    :param test_len: test length,
    :return:
        - the detrended datasets
        - the trend values that has to be added back after computing the prediction
    """
    # Compute mean values for hour/minute each day of the week (STATS ARE COMPUTED USING ONLY TRAIN SET)
    dt_idx = pd.DatetimeIndex(df[DATETIME])
    df_copy = df.set_index(dt_idx, drop=False)
    df_train_mean = \
        df_copy.iloc[:train_len].groupby(
            [df_copy.iloc[:train_len].index.hour,
             df_copy.iloc[:train_len].index.minute])[TARGET].mean()
    # Remove mean values from dataset
    df_copy['trend'] = None
    for h in df_train_mean.index.levels[0]:
        for m in df_train_mean.index.levels[1]:
            mu = df_train_mean[h, m]
            idxs = df_copy.loc[(df_copy.index.hour == h)
                               & (df_copy.index.minute == m)].index
            df_copy.loc[idxs,
                        TARGET] = df_copy.loc[idxs,
                                              TARGET].apply(lambda x: x - mu)
            df_copy.loc[idxs, 'trend'] = mu
    df[TARGET] = df_copy[TARGET].values
    return df, np.float32(df_copy['trend'].values)
