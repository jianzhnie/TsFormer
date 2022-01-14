import argparse
import os
from datetime import datetime

import numpy as np
import pandas as pd

NAME = 'uci'
FREQ = '15T'
DATETIME = 'date'
TARGET = 'Global_active_power'

parser = argparse.ArgumentParser(description='Data Preprocess')
parser.add_argument(
    '--data_dir',
    type=str,
    required=False,
    default='/home/wenqi-ao/userdata/workdirs/data/power_load/house_power',
    help='Directory of data.')
parser.add_argument(
    '--fill_nan',
    type=str,
    required=False,
    default='median',
    help='Method to fill nan values')

args = parser.parse_args()


def set_datetime_index(df, datetime_col='datetime'):
    if not isinstance(df.index, pd.core.indexes.datetimes.DatetimeIndex):
        try:
            dt_idx = pd.DatetimeIndex(df[datetime_col])
            df = df.set_index(dt_idx, drop=False)
            return df
        except ValueError:
            raise ValueError(
                '{0} is not the correct datetime column name or the column values '
                'are not formatted correctly (use datetime)'.format(
                    datetime_col))
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


def process_csv(args):
    """Parse the datetime field, Sort the values accordingly and save the new
    dataframe to disk."""
    df = pd.read_csv(
        os.path.join(args.data_dir, 'household_power_consumption.txt'),
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
    df = impute_missing(
        df, method=args.fill_nan, values_col=TARGET, datetime_col=DATETIME)
    print(df.head(5))
    print(df.tail(5))
    print(df.info())
    df.to_csv(
        os.path.join(args.data_dir,
                     'UCI_household_power_consumption_synth.csv'),
        index=False)


if __name__ == '__main__':
    process_csv(args)
