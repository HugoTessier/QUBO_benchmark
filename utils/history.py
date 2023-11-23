from typing import Any, Tuple, List

__all__ = ['History', 'OLS', 'OUTERMOST_LOOP_STEP', 'ILS', 'INNERMOST_LOOP_STEP', 'ENERGY', 'INSTANT', 'SUM']

# Constants for convenience
OLS = "OUTERMOST_LOOP_STEP"
OUTERMOST_LOOP_STEP = OLS

ILS = "INNERMOST_LOOP_STEP"
INNERMOST_LOOP_STEP = ILS
ENERGY = "ENERGY"

INSTANT = "INSTANT"
SUM = "SUM"


class History:
    """
    Class that records entries and is able to process them to provide plots.

    Some examples of entries (to help uniformize between different algorithms):
        "outermost_loop_step": to increment for any Monte-Carlo/Euler/Gradient Descent step, i.e. the main loop.
        "innermost_loop_step": to increment at any loop that has to be run sequentially and cannot be parallelized.
        "energy": to monitor the evolution of the solution's energy.

    Other possible entries are operations, that are defined in utils.operations.
    """

    # Said examples are proposed as constants right up there.

    def __init__(self):
        self.content: List[Tuple] = []

    def record(self, key: str, value: Any = 1) -> None:
        """
        Record an entry.

        :param key: Type of the entry, e.g.: "step", "energy", "FLOPs"...
        :param value: Value of the entry (default: 1, so that it corresponds to an increment).
        """
        self.content.append((key, value))

    def plot(self, x_key: List[str], y_key: List[str], x_mode: str = SUM,
             y_mode: str = INSTANT, x_weights: dict = None, y_weights: dict = None) -> Tuple[List, List]:
        """
        Generates a list of coordinates values of y as a function of x.
        x and y have to possible modes:
            1) "instant": considers the last value encountered.
            2) "sum": considers the sum of all previously encountered entries.
        The default values provide an example: if we have the x_key corresponding to a step increase, then x_mode="sum"
        means we plot the progression across elapsed steps; if the y_key is some sort of energy, then y_mode="instant"
        means that, for each step incrementation, we give the first next recorded energy (or the last one if there are
        some more steps recorded after the last energy entry).
        Having the y_mode="sum" could be used, for example, to plot the total mass of operations executed since the
        beginning for each step.

        The weights can be used to give a different importance to every key in x or y. For example, if x is a list of
        possible operations, such as addition or multiplication, it is possible to tell if we want multiplications to
        cost more than addition, which will then be accounted in the produced plot.

        :param x_key: The key elements to consider as the x.
        :param y_key: The key elements to consider as the y.
        :param x_mode: "instant" or "sum"
        :param y_mode: "instant" or "sum"
        :param x_weights: Weights of each type of element in x.
        :param y_weights: Weights of each type of element in y.
        :return: x and y as two lists.
        """
        x = []
        y = []
        x_temp = 0
        y_temp = 0
        x_to_fill = []
        for k, v in self.content:
            if k in x_key:
                if x_weights is not None:  # Apply weight
                    if k in x_weights:
                        v *= x_weights[k]
                if x_mode == INSTANT:  # In both cases, we prepare the room for the next y we will put there
                    x.append(v)
                    y.append(None)
                    x_to_fill.append(len(y) - 1)
                elif x_mode == SUM:
                    x_temp += v
                    x.append(x_temp)
                    y.append(None)
                    x_to_fill.append(len(y) - 1)
                else:
                    raise ValueError
            if k in y_key:
                if y_weights is not None:  # Apply weight
                    if k in y_weights:
                        v *= y_weights[k]
                if y_mode == INSTANT:
                    for i in x_to_fill:
                        y[i] = v
                    if len(x_to_fill) == 0:  # If nowhere to be inserted, then it means we're still in the previous x
                        x.append(x_temp)
                        y.append(v)
                elif y_mode == SUM:
                    y_temp += v
                    for i in x_to_fill:
                        y[i] = y_temp
                    if len(x_to_fill) == 0:
                        x.append(x_temp)
                        y.append(y_temp)
                else:
                    raise ValueError
                x_to_fill = []

        # If there x after the last y, then we replace the last None values in y by the last valid y value
        y_temp = None
        for i, v in enumerate(y):
            if v is None:
                y[i] = y_temp
            else:
                y_temp = v
        return x, y
