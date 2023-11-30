"""
List of commonplace operations, with their history ID and a cost.
Allows uniformizing cost measurement across algorithms.
"""
import numpy as np
from utils.history import *

# Float-related operations
MULTIPLICATION = 'MULTIPLICATION'
DIVISION = 'DIVISION'
ADDITION = 'ADDITION'
SIGN = 'SIGN'
SIGN_FLIP = 'SIGN_FLIP'
EXP = 'EXP'
COMPARISON = 'COMPARISON'

# Other operations
RANDOM_NUMBER_GENERATION = 'RANDOM_NUMBER_GENERATION'
SPIN_FLIP = 'SPIN_FLIP'
VALUE_CHECK = 'VALUE_CHECK'
BITWISE_OR = 'BITWISE_OR'
CONDITIONAL_FILL = 'CONDITIONAL_FILL'
CLIP = 'CLIP'

ALL_OPERATIONS = [MULTIPLICATION,
                  DIVISION,
                  ADDITION,
                  SIGN,
                  SIGN_FLIP,
                  EXP,
                  COMPARISON,
                  RANDOM_NUMBER_GENERATION,
                  SPIN_FLIP,
                  VALUE_CHECK,
                  BITWISE_OR,
                  CONDITIONAL_FILL,
                  CLIP]
"""List of all operations."""

OPERATIONS_WEIGHTS = {
    MULTIPLICATION: 1.,
    DIVISION: 1.,
    ADDITION: 1.,
    SIGN: 1.,
    SIGN_FLIP: 1.,
    EXP: 1.,
    COMPARISON: 1.,
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
                self.history.record(MULTIPLICATION, a.shape[0] * a.shape[1] * b.shape[1])
                self.history.record(ADDITION, a.shape[0] * (a.shape[1] - 1) * b.shape[1])
            else:  # len(a.shape) == 1
                self.history.record(MULTIPLICATION, a.shape[0] * a.shape[1])
                self.history.record(ADDITION, a.shape[0] * (a.shape[1] - 1))
        else:  # len(a.shape) == 1:
            if len(b.shape) == 2:
                self.history.record(MULTIPLICATION, b.shape[0] * b.shape[1])
                self.history.record(ADDITION, (b.shape[0] - 1) * b.shape[1])
            else:  # len(a.shape) == 1
                self.history.record(MULTIPLICATION, a.shape[0])
                self.history.record(ADDITION, a.shape[0] - 1)

    def addition(self, amount: int = 1):
        self.history.record(ADDITION, amount)

    def subtraction(self, amount: int = 1):
        self.history.record(SIGN_FLIP, amount)
        self.history.record(ADDITION, amount)

    def sign(self, amount: int = 1):
        self.history.record(SIGN, amount)

    def sign_flip(self, amount: int = 1):
        self.history.record(SIGN_FLIP, amount)

    def multiplication(self, amount: int = 1):
        self.history.record(MULTIPLICATION, amount)

    def division(self, amount: int = 1):
        self.history.record(DIVISION, amount)

    def exp(self, amount: int = 1):
        self.history.record(EXP, amount)

    def comparison(self, amount: int = 1):
        self.history.record(EXP, amount)

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
           'OPERATIONS_WEIGHTS',
           *ALL_OPERATIONS]
