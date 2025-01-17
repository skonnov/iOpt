from __future__ import annotations

import copy
import sys

import numpy as np
from depq import DEPQ

import json

from iOpt.problem import Problem
from iOpt.solution import Solution
from iOpt.trial import Point, FunctionValue, FunctionType
from iOpt.trial import Trial


# from bintrees import AVLTree


class SearchDataItem(Trial):
    """
        Класс SearchDataItem предназначен для хранения поисковой информации, представляющей собой
        интервал с включенной правой точкой, а так же ссылками на соседние интервалы. SearchDataItem
        является наследником от класса Trial.
    """

    def __init__(self, y: Point, x: np.double,
                 function_values: np.ndarray(shape=(1), dtype=FunctionValue) = [FunctionValue()],
                 discrete_value_index: int = 0):
        """
        Конструктор класса SearchDataItem

        :param y: Точка испытания в исходной N-мерной области поиска
        :param x: Отображении точки испытания y на отрезок [0, 1]
        :param function_values: Вектор значений функций (целевой функции и функций ограничений)
        :param discrete_value_index: Дискретный параметр
        """
        super().__init__(point=y, function_values=copy.deepcopy(function_values))
        self.point = y
        self.__x = x
        self.__discrete_value_index = discrete_value_index
        self.__index: int = -2
        self.__z: np.double = sys.float_info.max
        self.__leftPoint: SearchDataItem = None
        self.__rightPoint: SearchDataItem = None
        self.delta: np.double = -1.0
        self.globalR: np.double = -1.0
        self.localR: np.double = -1.0
        self.iterationNumber: int = -1

    def get_x(self) -> np.double:
        """
        Метод позволяет получить правую точку поискового интервала, где :math:`x\in[0, 1]`.

        :return: Значение правой точки интервала
        """
        return self.__x

    def get_y(self) -> Point:
        """
        Метод позволяет получить N-мерную точку испытания исходной области поиска.

        :return: Значение N-мерной точки испытания
        """
        return self.point

    def get_discrete_value_index(self) -> int:
        """
        Метод позволяет получить дискретный параметр.

        :return: Значение дискретного параметра
        """
        return self.__discrete_value_index

    def set_index(self, index: int):
        """
        Метод позволяет задать значение индекса последнего выполненного ограничения
        для индексной схемы.

        :param index: Индекс ограничения
        """
        self.__index = index

    def get_index(self) -> int:
        """
        Метод позволяет получить значение индекса последнего выполненного ограничения
        для индексной схемы.

        :return: Значение индекса
        """
        return self.__index

    def set_z(self, z: np.double):
        """
        Метод позволяет задать значение функции для заданного индекса.

        :param z: Значение функции
        """
        self.__z = z

    def get_z(self) -> np.double:
        """
        Метод позволяет получить значение функции для заданного индекса.

        :return: Значение функции для index
        """
        return self.__z

    def set_left(self, point: SearchDataItem):
        """
        Метод позволяет задать левый интервал для исходного.

        :param point: Левый интервал
        """
        self.__leftPoint = point

    def get_left(self) -> SearchDataItem:
        """
        Метод позволяет получить левый интервал для исходного.

        :return: Значение левого интервала
        """
        return self.__leftPoint

    def set_right(self, point: SearchDataItem):
        """
        Метод позволяет задать правый интервал для исходного.

        :param point: Правый интервал
        """
        self.__rightPoint = point

    def get_right(self) -> SearchDataItem:
        """
       Метод позволяет получить правый интервал для исходного.

       :return: Значение правого интервала
       """
        return self.__rightPoint

    def __lt__(self, other) -> bool:
        """
        Метод переопределяет оператор сравнения < для двух интервалов.
        :param other: Второй интервал
        :return: Значение true - если правая точка исходного интервала меньше
        правой точки второго, иначе - false.
        """
        return self.get_x() < other.get_x()


class CharacteristicsQueue:
    """
    Класс CharacteristicsQueue предназначен для хранения приоритетной очереди
    характеристик с вытеснением.
    """

    def __init__(self, maxlen: int):
        """
        Конструктор класса CharacteristicsQueue

        :param maxlen: Максимальный размер очереди
        """
        self.__baseQueue = DEPQ(iterable=None, maxlen=maxlen)

    def Clear(self):
        """
        Метод позволяет очистить очередь
        """
        self.__baseQueue.clear()

    def insert(self, key: np.double, data_item: SearchDataItem):
        """
        Метод добавляет поисковый интервал с указанным приоритетом.
        Приоритетом является значение характеристики на данном интервале.

        :param key: Приоритет поискового интервала
        :param data_item: Вставляемый интервал
        """
        self.__baseQueue.insert(data_item, key)

    def get_best_item(self) -> (SearchDataItem, np.double):
        """
        Метод позволяет получить интервал с лучшей характеристикой

        :return: Кортеж: интервал с лучшей характеристикой, приоритет интервала в очереди
        """
        return self.__baseQueue.popfirst()

    def is_empty(self):
        """
        Метод позволяет сделать проверку на пустоту очереди.

        :return: Значение true если очередь пуста, иначе false
        """
        return self.__baseQueue.is_empty()

    def get_max_len(self) -> int:
        """
        Метод позволяет получить максимальный размер очереди.

        :return: Значение максимального размера очереди
        """
        return self.__baseQueue.maxlen

    def get_len(self) -> int:
        """
        Метод позволяет получить текущий размер очереди.

        :return: Значение текущего размера очереди
        """
        return len(self.__baseQueue)


class SearchData:
    """
    Класс SearchData предназначен для хранения множества всех интервалов, исходной задачи
    и приоритетной очереди глобальных характеристик.
    """

    # очереди характеристик
    # _RGlobalQueue: CharacteristicsQueue = CharacteristicsQueue(None)
    # упорядоченное множество всех испытаний по X
    # __allTrials: AVLTree = AVLTree()
    # _allTrials: List = []
    # __firstDataItem:

    # solution: Solution = None

    def __init__(self, problem: Problem, maxlen: int = None):
        """
        Конструктор класса SearchData

        :param problem: Информация об исходной задаче
        :param maxlen: Максимальный размер очереди
        """
        self.solution = Solution(problem)
        self._allTrials = []
        self._RGlobalQueue = CharacteristicsQueue(maxlen)
        self.__firstDataItem: SearchDataItem = None

    def clear_queue(self):
        """
        Метод позволяет очистить очередь характеристик
        """
        self._RGlobalQueue.Clear()

    # вставка точки если знает правую точку
    # в качестве интервала используем [i-1, i]
    # если right_data_item == None то его необходимо найти по дереву _allTrials
    def insert_data_item(self, new_data_item: SearchDataItem,
                         right_data_item: SearchDataItem = None):
        """
        Метод позволяет добавить новый интервал испытаний в список всех проведенных испытаний
        и приоритетную очередь характеристик.

        :param new_data_item: Новый интервал испытаний
        :param right_data_item: Покрывающий интервал, является правым интервалом для newDataItem
        """
        flag = True
        if right_data_item is None:
            right_data_item = self.find_data_item_by_one_dimensional_point(new_data_item.get_x())
            flag = False
        new_data_item.set_left(right_data_item.get_left())
        right_data_item.set_left(new_data_item)
        new_data_item.set_right(right_data_item)
        new_data_item.get_left().set_right(new_data_item)

        self._allTrials.append(new_data_item)

        self._RGlobalQueue.insert(new_data_item.globalR, new_data_item)
        if flag:
            self._RGlobalQueue.insert(right_data_item.globalR, right_data_item)

    def insert_first_data_item(self, left_data_item: SearchDataItem,
                               right_data_item: SearchDataItem):
        """
        Метод позволяет добавить пару интервалов испытаний на первой итерации AGP.

        :param left_data_item: Левый интервал для right_data_item
        :param right_data_item: Правый интервал для leftDataItem
        """
        left_data_item.set_right(right_data_item)
        right_data_item.set_left(left_data_item)

        self._allTrials.append(left_data_item)
        self._allTrials.append(right_data_item)

        self.__firstDataItem = left_data_item

    # поиск покрывающего интервала
    # возвращает правую точку
    def find_data_item_by_one_dimensional_point(self, x: np.double) -> SearchDataItem:
        """
        Метод позволяет найти покрывающий интервал для полученной точки x.

        :param x: Правая точка интервала
        :return: Правая точка покрывающего интервала
        """
        # итерируемся по rightPoint от минимального элемента
        for item in self:
            if item.get_x() > x:
                return item
        return None

    def get_data_item_with_max_global_r(self) -> SearchDataItem:
        """
        Метод позволяет получить интервал с лучшим значением глобальной характеристики.

        :return: Значение интервала с лучшей глобальной характеристикой
        """
        if self._RGlobalQueue.is_empty():
            self.refill_queue()
        return self._RGlobalQueue.get_best_item()[0]

    # Перезаполнение очереди (при ее опустошении или при смене оценки константы Липшица)
    def refill_queue(self):
        """
        Метод позволяет перезаполнить очередь глобальных характеристик, например, при ее опустошении
        или при смене оценки константы Липшица.

        """
        self._RGlobalQueue.Clear()
        for itr in self:
            self._RGlobalQueue.insert(itr.globalR, itr)

    # Возвращает текущее число интервалов в дереве
    def get_count(self) -> int:
        """
        Метод позволяет получить текущее число интервалов в списке.

        :return: Значение числа интервалов в списке
        """
        return len(self._allTrials)

    def get_last_item(self) -> SearchDataItem:
        """
        Метод позволяет получить последний добавленный интервал в список.

        :return: Значение последнего добавленного интервала
        """
        try:
            return self._allTrials[-1]
        except Exception:
            print("GetLastItem: List is empty")

    def get_last_items(self, N: int = 1) -> list[SearchDataItem]:
        """
        Метод позволяет получить последние добавленные интервалы в список.

        :return: Значения последней серии добавленных интервалов
        """
        try:
            return self._allTrials[-N:]
        except Exception:
            print("GetLastItems: List is empty")

    def save_progress(self, file_name: str):
        """
        Сохранение процесса оптимизации в файл

        :param file_name: имя файла
        """
        data = {}
        data['SearchDataItem'] = []
        for dataItem in self._allTrials:

            fvs = []
            for fv in dataItem.function_values:
                fvs.append({
                    'value': fv.value,
                    'type': 1 if fv.type == FunctionType.OBJECTIV else 2,
                    'functionID': str(fv.functionID),
                })

            data['SearchDataItem'].append({
                'float_variables': list(dataItem.get_y().float_variables),
                'discrete_variables': [] if dataItem.get_y().discrete_variables is None else list(
                    dataItem.get_y().discrete_variables),
                'function_values': list(fvs),
                'x': dataItem.get_x(),
                'delta': dataItem.delta,
                'globalR': dataItem.globalR,
                'localR': dataItem.localR,
                'index': dataItem.get_index(),
                'discrete_value_index': dataItem.get_discrete_value_index(),
                '__z': dataItem.get_z()
            })

        data['best_trials'] = []  # создаем список
        dataItem = self.solution.best_trials[0]
        fvs = []  # пустой список для словарей со значениями функций
        for fv in dataItem.function_values:
            fvs.append({
                'value': fv.value,
                'type': 1 if fv.type == FunctionType.OBJECTIV else 2,
                'functionID': str(fv.functionID),
            })

        data['best_trials'].append({
            'float_variables': list(dataItem.get_y().float_variables),
            'discrete_variables': [] if dataItem.get_y().discrete_variables is None else list(
                dataItem.get_y().discrete_variables),
            'function_values': list(fvs),
            'x': dataItem.get_x(),
            'delta': dataItem.delta,
            'globalR': dataItem.globalR,
            'localR': dataItem.localR,
            'index': dataItem.get_index(),
            'discrete_value_index': dataItem.get_discrete_value_index(),
            '__z': dataItem.get_z()
        })

        with open(file_name, 'w') as f:
            json.dump(data, f, indent='\t', separators=(',', ':'))

    def load_progress(self, file_name: str):
        """
        Загрузка процесса оптимизации из файла

        :param file_name: имя файла
        """

        with open(file_name) as json_file:
            data = json.load(json_file)

            function_values = []
            for p in data['best_trials']:

                for fv in p['function_values']:
                    function_values.append(FunctionValue(
                        (FunctionType.OBJECTIV if fv['type'] == 1 else FunctionType.CONSTRAINT),
                        str(fv['functionID'])))
                    function_values[-1].value = np.double(fv['value'])

                data_item = SearchDataItem(Point(p['float_variables'], p['discrete_variables']), p['x'],
                                           function_values,
                                           p['discrete_value_index'])
                data_item.delta = p['delta']  # [-1] - обращение к последнему элементу
                data_item.globalR = p['globalR']
                data_item.localR = p['localR']
                data_item.set_z(p['__z'])
                data_item.set_index(p['index'])

                self.solution.best_trials[0] = data_item

            first_data_item = []

            for p in data['SearchDataItem'][:2]:
                function_values = []

                for fv in p['function_values']:
                    function_values.append(FunctionValue(
                        (FunctionType.OBJECTIV if fv['type'] == 1 else FunctionType.CONSTRAINT),
                        str(fv['functionID'])))
                    function_values[-1].value = np.double(fv['value'])

                first_data_item.append(
                    SearchDataItem(Point(p['float_variables'], p['discrete_variables']), p['x'], function_values,
                                   p['discrete_value_index']))
                first_data_item[-1].delta = p['delta']
                first_data_item[-1].globalR = p['globalR']
                first_data_item[-1].localR = p['localR']
                first_data_item[-1].set_index(p['index'])

            self.insert_first_data_item(first_data_item[0], first_data_item[1])

            for p in data['SearchDataItem'][2:]:
                function_values = []

                for fv in p['function_values']:
                    function_values.append(FunctionValue(
                        (FunctionType.OBJECTIV if fv['type'] == 1 else FunctionType.CONSTRAINT),
                        str(fv['functionID'])))
                    function_values[-1].value = np.double(fv['value'])

                data_item = SearchDataItem(Point(p['float_variables'], p['discrete_variables']),
                                           p['x'], function_values, p['discrete_value_index'])
                data_item.delta = p['delta']
                data_item.globalR = p['globalR']
                data_item.localR = p['localR']
                data_item.set_z(p['__z'])
                data_item.set_index(p['index'])

                self.insert_data_item(data_item)

    def __iter__(self):
        # вернуть самую левую точку из дерева (ниже код проверить!)
        # return self._allTrials.min_item()[1]
        self.curIter = self.__firstDataItem
        if self.curIter is None:
            raise StopIteration
        else:
            return self

    def __next__(self):
        if self.curIter is None:
            raise StopIteration
        else:
            tmp = self.curIter
            self.curIter = self.curIter.get_right()
            return tmp


class SearchDataDualQueue(SearchData):
    """
    Класс SearchDataDualQueue является наследником класса SearchData. Предназначен
      для хранения множества всех интервалов, исходной задачи и двух приоритетных очередей
      для глобальных и локальных характеристик.

    """

    def __init__(self, problem: Problem, maxlen: int = None):
        """
        Конструктор класса SearchDataDualQueue

        :param problem: Информация об исходной задаче
        :param maxlen: Максимальный размер очереди
        """
        super().__init__(problem, maxlen)
        self.__RLocalQueue = CharacteristicsQueue(maxlen)

    def clear_queue(self):
        """
        Метод позволяет очистить очереди характеристик
        """
        self._RGlobalQueue.Clear()
        self.__RLocalQueue.Clear()

    def insert_data_item(self, new_data_item: SearchDataItem,
                         right_data_item: SearchDataItem = None):
        """
        Метод позволяет добавить новый интервал испытаний в список всех проведенных испытаний
          и приоритетные очереди глобальных и локальных характеристик.

        :param new_data_item: Новый интервал испытаний
        :param right_data_item: Покрывающий интервал, является правым интервалом для newDataItem
        """
        flag = True
        if right_data_item is None:
            right_data_item = self.find_data_item_by_one_dimensional_point(new_data_item.get_x())
            flag = False

        new_data_item.set_left(right_data_item.get_left())
        right_data_item.set_left(new_data_item)
        new_data_item.set_right(right_data_item)
        new_data_item.get_left().set_right(new_data_item)

        self._allTrials.append(new_data_item)

        self._RGlobalQueue.insert(new_data_item.globalR, new_data_item)
        self.__RLocalQueue.insert(new_data_item.localR, new_data_item)
        if flag:
            self._RGlobalQueue.insert(right_data_item.globalR, right_data_item)
            self.__RLocalQueue.insert(right_data_item.localR, right_data_item)

    def get_data_item_with_max_global_r(self) -> SearchDataItem:
        """
       Метод позволяет получить интервал с лучшим значением глобальной характеристики.

       :return: Значение интервала с лучшей глобальной характеристикой
       """
        if self._RGlobalQueue.is_empty():
            self.refill_queue()
        best_item = self._RGlobalQueue.get_best_item()
        while best_item[1] != best_item[0].globalR:
            if self._RGlobalQueue.is_empty():
                self.refill_queue()
            best_item = self._RGlobalQueue.get_best_item()
        return best_item[0]

    def get_data_item_with_max_local_r(self) -> SearchDataItem:
        """
       Метод позволяет получить интервал с лучшим значением локальной характеристики.

       :return: Значение интервала с лучшей локальной характеристикой
       """
        if self.__RLocalQueue.is_empty():
            self.refill_queue()
        best_item = self.__RLocalQueue.get_best_item()
        while best_item[1] != best_item[0].localR:
            if self.__RLocalQueue.is_empty():
                self.refill_queue()
            best_item = self.__RLocalQueue.get_best_item()
        return best_item[0]

    def refill_queue(self):
        """
       Метод позволяет перезаполнить очереди глобальных и локальных характеристик, например,
         при их опустошении или при смене оценки константы Липшица.

       """
        self.clear_queue()
        for itr in self:
            self._RGlobalQueue.insert(itr.globalR, itr)
            self.__RLocalQueue.insert(itr.localR, itr)
