'''
Author: jianzhnie
Date: 2022-01-14 14:08:40
LastEditTime: 2022-01-14 15:02:44
LastEditors: jianzhnie
Description:

'''
import argparse
import os

import pandas as pd

NAME = 'gefcom'
FREQ = 'H'
DATETIME = 'date'

parser = argparse.ArgumentParser(description='Data Preprocess')
parser.add_argument(
    '--data_dir',
    type=str,
    required=False,
    default='/home/wenqi-ao/userdata/workdirs/data/power_load/GEFCom2014Data',
    help='Directory of data.')

args = parser.parse_args()


def process_csv(args):
    """Parse the datetime field, Sort the values accordingly and save the new
    dataframe to disk."""
    df = load_raw_dataset(args)
    df.to_csv(os.path.join(args.data_dir, 'gefcom2014.csv'), index=False)
    print(df.head(5))
    print(df.info())


def load_raw_dataset(args):
    """Load the dataset as is.

    :return: pandas.DataFrame: sorted dataframe with parsed datetime
    """
    data_dir = os.path.join(args.data_dir, 'Task 1')
    df = pd.read_csv(os.path.join(data_dir, 'L1-train.csv'))
    for i in range(2, 16):
        data_dir = os.path.join(args.data_dir, 'Task {}'.format(i))
        tmp = pd.read_csv(os.path.join(data_dir, 'L{}-train.csv'.format(i)))
        df = pd.concat([df, tmp], axis=0)
    df[DATETIME] = pd.date_range('01-01-2001', '12-01-2011', freq=FREQ)[1:]
    df = df[~pd.isnull(df.LOAD)].reset_index(drop=True)
    cols = [DATETIME] + ['w' + str(i) for i in range(1, 26)] + ['LOAD']
    df = df[cols]
    return df


if __name__ == '__main__':
    process_csv(args)
