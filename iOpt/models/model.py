from abc import ABC, abstractmethod
from iOpt.method.search_data import SearchDataItem


class Model(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def fit(self, X: list, y: list):
        pass

    @abstractmethod
    def calculate_dot_characteristic(self, *point):
        pass

    @abstractmethod
    def calculate_r_ps(self, curr_point: SearchDataItem, left_point: SearchDataItem):
        pass

    @abstractmethod
    def get_model(self):
        pass