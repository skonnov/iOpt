from abc import ABC, abstractmethod
from iOpt.method.search_data import SearchDataItem


class Model:
    def __init__(self):
        pass

    # @abstractmethod
    def fit(self, X: list, y: list):
        pass

    # @abstractmethod
    def calculate_r_ps(self, curr_point: SearchDataItem, left_point: SearchDataItem):
        pass