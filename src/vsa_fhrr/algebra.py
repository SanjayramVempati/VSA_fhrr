import cupy as cp
from numpy.typing import NDArray

class FHRR:

    def __init__(self, dim: int) -> None:
        self.dim = dim

    def __repr__(self):
        return f"FHRR(dim={self.dim})"

    def zero(self) -> NDArray[cp.complex64]:
        """
        Returns the addative/bundling identity vector (zero vector) for FHRR.
        """
        return cp.zeros(self.dim, dtype=cp.complex64)
    
    def one(self) -> NDArray[cp.complex64]:
        """
        Returns the multiplicative identity vector (one vector) for FHRR.
        """
        return cp.ones(self.dim, dtype=cp.complex64)
    
    def random(self) -> NDArray[cp.complex64]:
        angles = cp.random.uniform(0, 2 * cp.pi, self.dim)
        return cp.exp(1j * angles)

    def bundle(self, a: NDArray[cp.complex64], b: NDArray[cp.complex64]) -> NDArray[cp.complex64]:
        """
        Bundles two FHRR vectors `a` and `b` by element-wise addition.
        """
        return a + b
        
    def bundles(self, vectors: NDArray[cp.complex64]) -> NDArray[cp.complex64]:
        return cp.sum(vectors, axis=0)
    

    def normalize(self, a: NDArray[cp.complex64]) -> NDArray[cp.complex64]:
        """
        Normalizes the FHRR vector `a` to have unit magnitude.
        """
        return cp.sqrt(self.dim) * a / cp.linalg.norm(a)


    def centroid(self, vectors: NDArray[cp.complex64]) -> NDArray[cp.complex64]:
        """
        Computes the centroid of a set of FHRR vectors by averaging them.
        """
        return self.normalize(FHRR.bundles(vectors))

    def bind(self, a: NDArray[cp.complex64], b: NDArray[cp.complex64]) -> NDArray[cp.complex64]:
        """
        Binds two FHRR vectors `a` and `b` by element-wise multiplication.
        """
        return a * b

    def binds(self, vectors: NDArray[cp.complex64]) -> NDArray[cp.complex64]:
        result = cp.ones(vectors.shape[1], dtype=cp.complex64)
        for vec in vectors:
            result *= vec
        return result
    
    
    def invert(self, a: NDArray[cp.complex64]) -> NDArray[cp.complex64]:
        """
        Inverts vector `a` by taking its complex conjugate.
        """
        return cp.conj(a)

    
    def unbind(self, a: NDArray[cp.complex64], b: NDArray[cp.complex64]) -> NDArray[cp.complex64]:
        """
        Unbinds vector `b` from vector `a` by multiplying `a` with the complex conjugate of `b`.
        """
        return a * cp.conj(b)
    
    
    def cosine_similarity(self, a: NDArray[cp.complex64], b: NDArray[cp.complex64]) -> float:
        """
        Computes the cosine similarity between two FHRR vectors.
        """
        dot_product = cp.vdot(a, b)
        norm_a = cp.linalg.norm(a)
        norm_b = cp.linalg.norm(b)
        return float(cp.real(dot_product) / (norm_a * norm_b))
    
    
    def dotprod_similarity(self, a: NDArray[cp.complex64], b: NDArray[cp.complex64]) -> float:
        """
        Computes the dot product similarity between two FHRR vectors.
        Assumes both vectors are unitary
        """
        return float(cp.real(cp.vdot(a, b))) / self.dim
    
    @staticmethod
    def permute(a: NDArray[cp.complex64], shifts: int) -> NDArray[cp.complex64]:
        """
        Permutes the vector `a` by circularly shifting it to the right by `shifts` positions.
        """
        return cp.roll(a, shifts)
    
    
    @staticmethod
    def fpe(base: NDArray[cp.complex64], x: float) -> NDArray[cp.complex64]:
        """
        Raises the FHRR vector `base` to the float `x` using element-wise exponentiation.
        """
        return cp.power(base, x)
    

    

