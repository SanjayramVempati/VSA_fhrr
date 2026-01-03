import cupy as cp
from numpy.typing import NDArray
from .algebra import FHRR

class FPE_Encoder:
    def __init__(self, dim: int) -> None:
        self.fhrr = FHRR(dim)
        self.dim = dim
        self.base_vector = self.fhrr.random()
    
    def encode_float(self, x: float) -> NDArray[cp.complex64]:
        return self.fhrr.fpe(self.base_vector, x)

    def decode(self, vec: NDArray[cp.complex64]) -> float:
        pass
