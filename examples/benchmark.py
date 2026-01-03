import os
import sys
import numpy as np
import time

try:
    from hdc_sanjay.src.algebra import FHRR
except ModuleNotFoundError:
    # If the package isn't on sys.path (running the script directly),
    # add the project root to sys.path so the package becomes importable.
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from hdc_sanjay.src.algebra import FHRR


fhrr = FHRR(dim=10000)

print(fhrr)

start = time.perf_counter()
for j in range(100000):
    vec = fhrr.random()
end = time.perf_counter()
print(f"GPU: Time to create 100,000 random vectors: {end - start} seconds")

start = time.perf_counter()
for j in range(100000):
    vec = np.exp(1j*np.random.uniform(0, 2*np.pi, fhrr.dim))
    
end = time.perf_counter()
print(f"CPU: Time to create 100,000 random vectors: {end - start} seconds")

