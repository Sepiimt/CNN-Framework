# --- Imports ---
from typing import Optional, Literal
import numpy as np

PaddingVar = Literal["Constant", "Reflect", "Reflect_101", "Replicate"]
DoubleIntTuple = tuple[int, int]
TripleIntTuple = tuple[int, int, int]
IntList = list[int]

ConvPaddingTypes = Literal['zeros', 'reflect', 'replicate', 'circular']
ConvAcFuncTypes = Literal["ReLU", "LeakyReLU"]
InputChannels = Literal[1, 2, 3]

