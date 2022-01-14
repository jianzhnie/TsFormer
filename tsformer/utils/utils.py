'''
Author: jianzhnie
Date: 2022-01-11 16:25:58
LastEditTime: 2022-01-14 17:19:30
LastEditors: jianzhnie
Description:

'''
from datetime import datetime, time

import numpy as np
import pandas as pd


def get_df_time_slice(df, hour, minute):
    t = time(hour, minute, 0)
    mask = df.date.apply(lambda x: x.to_pydatetime().time()) == t
    return df[mask]


def shuffle_x_y(X, y):
    idxs = np.arange(X.shape[0])
    np.random.shuffle(idxs)
    return X[idxs], y[idxs]


def split_on_date(df, split_date='2007/1/1'):
    df = df.sort_values('date').reset_index()
    split_pt = min(df[df['date'] == datetime.strptime(
        split_date, '%Y/%m/%d').date()].index)
    return df.iloc[:split_pt], df.iloc[split_pt:]


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
