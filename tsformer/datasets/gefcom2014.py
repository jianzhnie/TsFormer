import io
import logging
import os
import warnings
import zipfile
from datetime import datetime

import numpy as np
import pandas as pd
import requests
from sklearn.preprocessing import MinMaxScaler, StandardScaler

logger = logging.getLogger('log')

NAME = 'gefcom'
SAMPLES_PER_DAY = 24
FREQ = 'H'
TARGET = 'LOAD'
DATETIME = 'date'
config = {'data': 'gefcom'}


def download():
    """
    Download data and extract them in the data directory
    :return:
    """
    url = 'https://www.dropbox.com/s/pqenrr2mcvl0hk9/GEFCom2014.zip?dl=1'
    logger.info('Donwloading .zip file from {}'.format(url))
    r = requests.get(url)
    logger.info('File downloaded.')
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(path=config['data'])

    new_loc = os.path.join(config['data'], 'GEFCom2014')
    os.rename(os.path.join(config['data'], 'GEFCom2014 Data'), new_loc)

    z = zipfile.ZipFile(os.path.join(new_loc, 'GEFCom2014-L_V2.zip'))
    z.extractall(path=os.path.join(new_loc))

    logger.info('File extracted and available at {}'.format(new_loc))
    process_csv()
    logger.info(
        'The original CSV has been parsed and is now available at {}'.format(
            os.path.join(config['data'],
                         'UCI_household_power_consumption_synth.csv')))


def process_csv(config):
    """Parse the datetime field, Sort the values accordingly and save the new
    dataframe to disk."""
    df = load_raw_dataset()
    cols = [DATETIME] + ['w'+ str(i) for i in range(1, 26)] + ['LOAD']
    df = df[cols]
    df.to_csv(
        os.path.join(config['data'], 'GEFCom2014/Load/gefcom2014.csv'),
        index=False)


def load_raw_dataset():
    """Load the dataset as is.

    :return: pandas.DataFrame: sorted dataframe with parsed datetime
    """
    data_dir = os.path.join(config['data'], 'GEFCom2014/Load/Task 1/')
    df = pd.read_csv(os.path.join(data_dir, 'L1-train.csv'))
    for i in range(2, 16):
        data_dir = os.path.join(config['data'],
                                'GEFCom2014/Load/Task {}/'.format(i))
        tmp = pd.read_csv(os.path.join(data_dir, 'L{}-train.csv'.format(i)))
        df = pd.concat([df, tmp], axis=0)
    df[DATETIME] = pd.date_range('01-01-2001', '12-01-2011', freq=FREQ)[1:]
    df = df[~pd.isnull(df.LOAD)].reset_index(drop=True)
    return df


def load_dataset():
    """Load an already cleaned version of the dataset."""
    df = pd.read_csv(
        os.path.join(config['data'], 'GEFCom2014/Load/gefcom2014.csv'))
    df[DATETIME] = df[DATETIME].apply(
        lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
    return df


def _add_holidays(df):
    """Add a binary variable to the dataset that takes value: 1 if the day is a
    holiday, 0 otherwise. Main holidays for the New England area are
    considered.

    :param df: the datafrme
    :return: the agumented dtaframe
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        idx = []
        idx.extend(
            df[df.day == 1][df.month == 1].index.tolist())  # new year's eve
        idx.extend(
            df[df.day == 4][df.month == 7].index.tolist())  # independence day
        idx.extend(
            df[df.day == 11][df.month == 11].index.tolist())  # veternas day
        idx.extend(
            df[df.day == 25][df.month == 12].index.tolist())  # christams
        df.loc[idx, 'holiday'] = 1
        return df


def transform(X, scaler=None, scaler_type=None):
    """Apply standard scaling to the input variables.

    :param X: the data
    :param scaler: the scaler to use, None if StandardScaler has to be used
    :return:
        scaler used
        X transformed using scaler
    """
    if scaler is None:
        if scaler_type == 'minmax':
            scaler = MinMaxScaler()
        else:
            scaler = StandardScaler()
        scaler.fit(X)
    return scaler, scaler.transform(X)


def inverse_transform(X, scaler, trend=None):
    """
    :param X: the data
    :param scaler: the scaler that have been used for transforming X
    :param trend: the trebd values that has been removed from X. None if no detrending has been used.
        It has to be the same dim. as X.
    :return:
        X with trend adds back and de-standardized
    """
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
    hour/weekday.

                        :param df: the dataset
    :param test_len: test length,
    :return:
        - the detrended datasets
        - the trend values that has to be added back after computing the prediction
    """
    # Compute mean values for each hour of every day of the week (STATS ARE COMPUTED USING ONLY TRAIN SET)
    dt_idx = pd.DatetimeIndex(df[DATETIME])
    df_copy = df.set_index(dt_idx, drop=False)
    df_train_mean = \
        df_copy.iloc[:train_len].groupby(
            [df_copy.iloc[:train_len].index.hour])[TARGET].mean()
    # Remove mean values from dataset
    df_copy['trend'] = None
    for h in df_train_mean.index:
        mu = df_train_mean[h]
        idxs = df_copy.loc[(df_copy.index.hour == h)].index
        df_copy.loc[idxs, TARGET] = df_copy.loc[idxs,
                                                TARGET].apply(lambda x: x - mu)
        df_copy.loc[idxs, 'trend'] = mu
    df[TARGET] = df_copy[TARGET].values
    return df, np.float32(df_copy['trend'].values[:-1])


if __name__ == '__main__':
    config = {'data': 'data'}
    process_csv(config)
