'''
Author: jianzhnie
Date: 2022-01-11 16:25:58
LastEditTime: 2022-01-14 17:21:05
LastEditors: jianzhnie
Description:

'''
import os
from datetime import datetime

from matplotlib import pyplot as plt


def plot(x, samples_per_day=96, save_at=None):
    """Plot all time-series contained in the given list in the same plot.
    Produce two plots:

        - The whole length of time-series
        - A smaller window of the first 4 days
    The time-series are suuposed to be aligned in time.

    :param x: list of time-series.
        Each time-series should be of shape (batch_size, sequenece_length) or (batch_size, sequenece_length, 1)
    :param samples_per_day: int
        Number of samples for one day (e.g. measurements every 15 minute, then samples_per_day=96)
    :param save_at: str
        if None, do nothing else save the image at the `save_at` location.
    :return:
    """
    n_series = len(x)
    today = datetime.today()

    for i in range(n_series):
        y = x[i][::samples_per_day].reshape(-1)
        plt.plot(y[:samples_per_day * 4])
    if save_at is not None:
        plt.savefig('{}_{}_SMALL'.format(save_at, today))
    plt.show()

    for i in range(n_series):
        y = x[i][::samples_per_day].reshape(-1)
        plt.plot(y)
    if save_at is not None:
        plt.savefig('{}_{}'.format(save_at, today))
    plt.show()


def plot_full(path, data, feature):
    """Plot Full Graph of Energy Dataset."""
    data.plot(y=feature, figsize=(16, 8))
    plt.xlabel('DateTime', fontsize=10)
    plt.xticks(rotation=45)
    plt.ylabel(feature, fontsize=10)
    plt.grid()
    plt.title('{} Energy Prediction'.format(feature))
    plt.savefig(os.path.join(path, '{} Energy Prediction.png'.format(feature)))
    plt.show()


def plot_pred_test(pred, actual, path, feature, model, step):
    """Plot Test set Prediction."""
    plt.figure(figsize=(10, 8))

    plt.plot(pred, label='Pred')
    plt.plot(actual, label='Actual')

    plt.xlabel('Time', fontsize=18)
    plt.ylabel('{}'.format(feature), fontsize=18)

    plt.legend(loc='best')
    plt.grid()

    plt.title(
        '{} Energy Prediction using {} and {}'.format(feature, model, step),
        fontsize=18)
    plt.savefig(
        os.path.join(
            path, '{} Energy Prediction using {} and {}.png'.format(
                feature, model, step)))
