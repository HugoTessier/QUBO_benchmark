from dataclasses import dataclass, field
import numpy as np
from abc import ABC

__all__ = ['ProblemData', 'QUBOData', 'IsingData']


@dataclass
class ProblemData(ABC):
    """
    Dataclass that stores the data that define a combinatorial problem.
    Contains an additional, generic "extra" dictionary that can be used for convenience.
    """
    extra: dict = field(init=False, default_factory=dict)


@dataclass
class QUBOData(ProblemData):
    """
    Contains the data of a QUBO problem, i.e.:
    1) The numpy array "Q" containing the coupling coefficients, and
    2) The float energy "offset".
    """
    Q: np.ndarray
    offset: float


@dataclass
class IsingData(ProblemData):
    """
    Contains the data of an Ising model, i.e.:
    1) The numpy array "h" containing the local magnetic field, and
    2) The numpy array "J" containing the coupling coefficients, and
    3) The float energy "offset".
    """
    h: np.ndarray
    J: np.ndarray
    offset: float
