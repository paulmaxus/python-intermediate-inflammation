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


class Person:
    """Class representing a person."""
    def __init__(self, name):
        self.name = name

class Observation:
    """Class representing a single observation."""
    def __init__(self, value, day):
        self.value = value
        self.day = day

class Patient(Person):
    """Class representing a single patient."""
    def __init__(self, name):
        super().__init__(name)
        self.observations = []
    
    def add_observation(self, value, day=None):
        """Add an observation to the patient's record."""
        if day is None:
            try:
                day = self.observations[-1].day + 1
            except IndexError:
                day = 0
        observation = Observation(value, day)
        self.observations.append(observation)
        return observation

class Doctor(Person):
    """Class representing a doctor."""
    def __init__(self, name):
        super().__init__(name)
        self.patients = set()

    def add_patient(self, patient):
        """Add a patient to the doctor's list of patients."""
        self.patients.add(patient)

    def remove_patient(self, patient):
        """Remove a patient from the doctor's list of patients."""
        self.patients.remove(patient)
