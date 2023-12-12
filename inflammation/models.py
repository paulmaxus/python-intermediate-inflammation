"""Module containing models representing patients and their data.

The Model layer is responsible for the 'business logic' part of the software.

Patients' data is held in an inflammation table (2D array) where each row contains 
inflammation data for a single patient taken over a number of days 
and each column represents a single day across all patients.
"""

import numpy as np


def load_csv(filename):
    """Load a Numpy array from a CSV.

    :param filename: Filename of CSV to load.
    :return: Numpy array containing the data from the CSV.
    """
    return np.loadtxt(fname=filename, delimiter=',')


def daily_mean(data):
    """Calculate the daily mean of a 2D inflammation data array.
    
    :param data: 2D Numpy array containing inflammation data.
    :return: 1D Numpy array of mean values for each day.
    """
    return np.mean(data, axis=0)


def daily_max(data):
    """Calculate the daily max of a 2D inflammation data array.
    
    :param data: 2D Numpy array containing inflammation data.
    :return: 1D Numpy array of max values for each day.
    """
    return np.max(data, axis=0)


def daily_min(data):
    """Calculate the daily min of a 2D inflammation data array.
    
    :param data: 2D Numpy array containing inflammation data.
    :return: 1D Numpy array of minimum values for each day.
    """
    return np.min(data, axis=0)


def daily_std(data):
    """Calculate the daily standard deviation of a 2D inflammation data array.
    
    :param data: 2D Numpy array containing inflammation data.
    :return: 1D Numpy array of standard deviations for each day.
    """
    return np.std(data, axis=0)

def daily_above_threshold(row_number, data, threshold):
    """Calculate the days above threshold of a 1D inflammation data array.

    :param row_number: Row number of patient data to calculate.
    :param data: 2D Numpy array containing inflammation data.
    :param threshold: Threshold to compare against.
    :return: 1D Numpy array of days above threshold.
    """
    return list(map(lambda x: x > threshold, data[row_number]))
