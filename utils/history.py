from typing import Any, Tuple, List, Dict
from abc import ABC, abstractmethod

__all__ = ['History', 'MAIN_LOOP', 'SEQUENCE', 'ENERGY', 'InstantParser', 'SumParser']

# Constants for convenience
MAIN_LOOP = "MAIN_LOOP"
SEQUENCE = "SEQUENCE"
ENERGY = "ENERGY"


class HistoryParser(ABC):
    """Reads history entries, sorts, weights and processes them."""

    def __init__(self, keys_weights_dict: Dict):
        self.keys_weights_dict = keys_weights_dict
        self.content = None
        self.new_value = False

    @abstractmethod
    def _parse_new_value(self, value):
        raise NotImplementedError

    def read_new_value(self, key, value):
        if key in self.keys_weights_dict:
            self._parse_new_value(value * self.keys_weights_dict[key])
            self.new_value = True

    def got_new_value(self):
        return self.new_value

    def get_value(self):
        self.new_value = False
        return self.content


class InstantParser(HistoryParser):
    """Just gives back the last read value."""

    def _parse_new_value(self, value):
        self.content = value


class SumParser(HistoryParser):
    """Gives the sum of all read entries."""

    def _parse_new_value(self, value):
        if self.content is None:
            self.content = 0
        self.content += value


class History:
    """
    Class that records entries and is able to process them to provide plots.

    Some examples of entries (to help uniformize between different algorithms):
        "MAIN_LOOP": to increment for any Monte-Carlo/Euler/Gradient Descent step, or else...
        "SEQUENCE": to increment everytime something cannot be parallelized with what comes before.
        "ENERGY": to monitor the evolution of the solution's energy.

    Other possible entries are operations, that are defined in utils.operations.
    """

    # Said examples are proposed as constants right up there.

    def __init__(self, keys_list: List[str] = None, reducer_key: str = None):
        """
        :param keys_list: List of keys to record (others are ignored).
        :param reducer_key: Key between each record of which to fuse other records of same type together.
        """
        self.keys_list = keys_list
        self.reducer_key = reducer_key
        self.content: List[Tuple] = []
        self.buffer = {}

    def record(self, key: str, value: Any = 1) -> None:
        """
        Record an entry.

        :param key: Type of the entry, e.g.: "step", "energy", "FLOPs"...
        :param value: Value of the entry (default: 1, so that it corresponds to an increment).
        """
        if self.reducer_key is not None:
            if key == self.reducer_key:
                for k, v in self.buffer.items():
                    self.content.append((k, v))
                self.buffer = {}
                self.content.append((key, value))
            else:
                if self.keys_list is not None:
                    if key not in self.keys_list:
                        return None
                if key in self.buffer:
                    self.buffer[key] += value
                else:
                    self.buffer[key] = value
        else:
            if self.keys_list is not None:
                if key not in self.keys_list:
                    return None
            self.content.append((key, value))

    def plot(self, x_parser: HistoryParser, y_parser: HistoryParser) -> Tuple[List, List]:
        """
        Generates a list of coordinates values of y as a function of x.
        These x and y are parsed from the history using the HistoryParser objects.

        :param x_parser: HistoryParser dedicated to x.
        :param y_parser: HistoryParser dedicated to y.
        :return: x and y as two lists.
        """
        x = []
        y = []
        last_x = None
        last_y = None
        for k, v in self.content:
            x_parser.read_new_value(k, v)
            y_parser.read_new_value(k, v)
            if x_parser.got_new_value() or y_parser.got_new_value():
                last_x = x_parser.get_value()
                last_y = y_parser.get_value()
            if last_x is None or last_y is None:
                continue
            else:
                if len(x) == 0 and len(y) == 0:
                    x.append(last_x)
                    y.append(last_y)
                else:
                    if last_x != x[-1] and last_y != y[-1]:
                        x.append(last_x)
                        y.append(last_y)

        return x, y
