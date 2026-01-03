import os
import sys
import numpy as np
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.algebra import FHRR


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

