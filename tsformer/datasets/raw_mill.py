import logging
import os
import numpy as np
import pandas as pd
from uci_single_households import impute_missing, hourly_aggregate

logger = logging.getLogger('log')

SAMPLES_PER_DAY = 96
FREQ = 'min'
TARGET = 'target1'
DATETIME = 'date'

colnames_map = {
    '21_SERVER_SERVER::C410211/C410211MFSG.U(A喂料量给定)': 'target',
    'datetime': 'date'
}

colnames_map = {
    'datetime': 'date',
    '21_SERVER_SERVER::C41041P/C41041P03F.OUT(A磨磨机压差)': 'feature1',
    '21_SERVER_SERVER::C410413/C410413MIIF.OUT(A磨磨机电流)': 'feature2',
    '21_SERVER_SERVER::C411411/C411411MIIF.OUT(A出磨斗提电流)': 'feature3',
    '21_SERVER_SERVER::C410211/C410211MFSG.U(A喂料量给定)': 'target1',
    '21_SERVER_SERVER::C410211/C410211MFIF.OUT(A喂料量反馈)': 'target2',
    '21_SERVER_SERVER::C41041T/C41041T02F.OUT(A磨出磨温度)': 'feature4',
    '21_SERVER_SERVER::C5408to12/C5410ZS01G.SP(A磨热风阀给定)': 'feature5',
    '21_SERVER_SERVER::C5408to12/C5410ZI01F.OUT(A磨热风阀反馈)': 'feature6',
    '21_SERVER_SERVER::C5408to12/C5412ZS02G3.SP(A磨冷风阀给定)': 'feature7',
    '21_SERVER_SERVER::C5408to12/C5412ZI02F3.OUT(A磨冷风阀反馈)': 'feature8',
    '21_SERVER_SERVER::C4104AIAO/C4104AO3.U(A研磨压力给定)': 'feature9',
    '21_SERVER_SERVER::C4104AIAO/C4104AI6.OUT(A研磨压力反馈)': 'feature10',
    '21_SERVER_SERVER::C4104AIAO/C4104AI1.OUT(A主减垂直振动)': 'feature11',
    '21_SERVER_SERVER::C4104AIAO/C4104AI2.OUT(A主减水平振动)': 'feature12',
    '21_SERVER_SERVER::C4104/C4104M11VEVB.OUT(A磨主减输入垂直振动)': 'feature13',
    '21_SERVER_SERVER::C4104/C4104M11LEVB.OUT(A磨主减输入水平振动)': 'feature14',
    '21_SERVER_SERVER::C4104AIAO/C4104AO1.U(A磨选粉机频率给定)': 'feature15',
    '21_SERVER_SERVER::C4104AIAO/C4104AI7.OUT(A磨选粉机频率反馈)': 'feature16',
    '21_SERVER_SERVER::C4107aZ/C4107aZS01G.SP(A磨循环风机风门开度给定)': 'feature17',
    '21_SERVER_SERVER::C4107aZ/C4107aZI01F.OUT(A磨循环风机风门开度反馈)': 'feature18',
    '21_SERVER_SERVER::C41041T/C41041T01F.OUT(A磨入口温度)': 'feature19',
    '21_SERVER_SERVER::C41041P/C41041P01F.OUT(A磨入口压力)': 'feature20',
    '21_SERVER_SERVER::C41041P/C41041P02F.OUT(A磨出口压力)': 'feature21'
}


def process_csv(config):
    """Parse the datetime field, Sort the values accordingly and save the new
    dataframe to disk."""
    df = pd.read_csv(os.path.join(config['data']), sep=',', encoding="gb18030")
    colnames = list(colnames_map.keys())
    df = df[colnames]
    df.rename(columns=colnames_map, inplace=True)
    df[DATETIME] = pd.to_datetime(df[DATETIME], utc=False)
    df = hourly_aggregate(df, freq=FREQ, datetime_col=DATETIME)

    def parse(x):
        try:
            return np.float64(x)
        except ValueError:
            return np.nan

    df = df[df[TARGET] > 0]

    df[TARGET] = df[TARGET].apply(lambda x: parse(x))
    df = impute_missing(
        df,
        method=config['fill_nan'],
        values_col=TARGET,
        datetime_col=DATETIME)
    return df


if __name__ == '__main__':
    data_dir = 'data/raw_milla/data1.csv'
    config = {'data': data_dir, 'fill_nan': 'median'}
    df = process_csv(config)
    df.to_csv('data/raw_milla/data1_process.csv', index=False)
