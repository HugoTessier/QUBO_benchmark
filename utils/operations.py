"""
List of commonplace operations, with their history ID and a cost.
Allows uniformizing cost measurement across algorithms.
"""
import numpy as np
from utils.history import *

# Float-related operations
FLOAT_MULTIPLICATION = 'FLOAT_MULTIPLICATION'
FLOAT_DIVISION = 'FLOAT_DIVISION'
FLOAT_ADDITION = 'FLOAT_ADDITION'
FLOAT_SIGN = 'FLOAT_SIGN'
FLOAT_SIGN_FLIP = 'FLOAT_SIGN_FLIP'
FLOAT_EXP = 'FLOAT_EXP'
FLOAT_COMPARISON = 'FLOAT_COMPARISON'
ALL_FLOAT_OPERATIONS = [FLOAT_MULTIPLICATION,
                        FLOAT_DIVISION,
                        FLOAT_ADDITION,
                        FLOAT_SIGN,
                        FLOAT_SIGN_FLIP,
                        FLOAT_EXP,
                        FLOAT_COMPARISON]
"""List of float-related operations."""

# Other operations
RANDOM_NUMBER_GENERATION = 'RANDOM_NUMBER_GENERATION'
SPIN_FLIP = 'SPIN_FLIP'
VALUE_CHECK = 'VALUE_CHECK'
BITWISE_OR = 'BITWISE_OR'
CONDITIONAL_FILL = 'CONDITIONAL_FILL'
CLIP = 'CLIP'
MISCELLANEOUS_OPERATIONS = [RANDOM_NUMBER_GENERATION, SPIN_FLIP, VALUE_CHECK, BITWISE_OR, CONDITIONAL_FILL, CLIP]
"""List of non-float-related operations."""

ALL_OPERATIONS = [*ALL_FLOAT_OPERATIONS, *MISCELLANEOUS_OPERATIONS]
"""List of all operations."""

OPERATIONS_WEIGHTS = {
    FLOAT_MULTIPLICATION: 1.,
    FLOAT_DIVISION: 1.,
    FLOAT_ADDITION: 1.,
    FLOAT_SIGN: 1.,
    FLOAT_SIGN_FLIP: 1.,
    FLOAT_EXP: 1.,
    FLOAT_COMPARISON: 1.,
    RANDOM_NUMBER_GENERATION: 1.,
    SPIN_FLIP: 1.,
    VALUE_CHECK: 1.,
    BITWISE_OR: 1.,
    CONDITIONAL_FILL: 1.,
    CLIP: 1.
}
"""
Dictionary containing, for each type of operations defined in this module, its associated cost.
This dictionary can be edited, either to add the cost of custom operations, or to edit the existing ones.
By default, costs are set to 1, but replacing these values by the estimation of their energy cost could provide
a (very rough) estimate of an algorithm's energy cost.
"""


class OperationsRecorder:
    """Object that allows recording in a History object many kinds of operations."""

    def __init__(self, history: History):
        self.history = history

    def dot_product(self, a: np.ndarray, b: np.ndarray):
        if len(a.shape) == 2:
            if len(b.shape) == 2:
                self.history.record(FLOAT_MULTIPLICATION, a.shape[0] * a.shape[1] * b.shape[1])
                self.history.record(FLOAT_ADDITION, a.shape[0] * (a.shape[1] - 1) * b.shape[1])
            else:  # len(a.shape) == 1
                self.history.record(FLOAT_MULTIPLICATION, a.shape[0] * a.shape[1])
                self.history.record(FLOAT_ADDITION, a.shape[0] * (a.shape[1] - 1))
        else:  # len(a.shape) == 1:
            if len(b.shape) == 2:
                self.history.record(FLOAT_MULTIPLICATION, b.shape[0] * b.shape[1])
                self.history.record(FLOAT_ADDITION, (b.shape[0] - 1) * b.shape[1])
            else:  # len(a.shape) == 1
                self.history.record(FLOAT_MULTIPLICATION, a.shape[0])
                self.history.record(FLOAT_ADDITION, a.shape[0] - 1)

    def float_addition(self, amount: int = 1):
        self.history.record(FLOAT_ADDITION, amount)

    def float_subtraction(self, amount: int = 1):
        self.history.record(FLOAT_SIGN_FLIP, amount)
        self.history.record(FLOAT_ADDITION, amount)

    def float_sign(self, amount: int = 1):
        self.history.record(FLOAT_SIGN, amount)

    def float_sign_flip(self, amount: int = 1):
        self.history.record(FLOAT_SIGN_FLIP, amount)

    def float_multiplication(self, amount: int = 1):
        self.history.record(FLOAT_MULTIPLICATION, amount)

    def float_division(self, amount: int = 1):
        self.history.record(FLOAT_DIVISION, amount)

    def float_exp(self, amount: int = 1):
        self.history.record(FLOAT_EXP, amount)

    def float_comparison(self, amount: int = 1):
        self.history.record(FLOAT_EXP, amount)

    def random_number_generation(self, amount: int = 1):
        self.history.record(RANDOM_NUMBER_GENERATION, amount)

    def spin_flip(self, amount: int = 1):
        self.history.record(SPIN_FLIP, amount)

    def value_check(self, amount: int = 1):
        self.history.record(VALUE_CHECK, amount)

    def bitwise_or(self, amount: int = 1):
        self.history.record(BITWISE_OR, amount)

    def conditional_fill(self, amount: int = 1):
        self.history.record(CONDITIONAL_FILL, amount)

    def clip(self, amount: int = 1):
        self.history.record(CLIP, amount)


__all__ = ['OperationsRecorder',
           'ALL_OPERATIONS',
           'ALL_FLOAT_OPERATIONS',
           'MISCELLANEOUS_OPERATIONS',
           'OPERATIONS_WEIGHTS',
           *ALL_OPERATIONS]
