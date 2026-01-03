import os
import sys
import numpy as np
import cupy as cp
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

#create 3 vectors similar to starting vector
hv_starting = fhrr.random()

hv1 = FHRR.bind(hv_starting, fhrr.fpe(fhrr.random(), 0.5))
hv2 = FHRR.bind(hv_starting, fhrr.fpe(fhrr.random(), 0.1))
hv3 = FHRR.bind(hv_starting, fhrr.fpe(fhrr.random(), 0.9))

#take similarities of each of the vectors to the starting vector
print(f"hv1 similarity to starting hv: {FHRR.cosine_similarity(hv1, hv_starting):.3f}")
print(f"hv2 similarity to starting hv: {FHRR.cosine_similarity(hv2, hv_starting):.3f}")
print(f"hv3 similarity to starting hv: {FHRR.cosine_similarity(hv3, hv_starting):.3f}")

#take similarities of each vector to eachother
print(f"hv1 similarity to hv2: {FHRR.cosine_similarity(hv1, hv2):.3f}")
print(f"hv2 similarity to hv3: {FHRR.cosine_similarity(hv2, hv3):.3f}")
print(f"hv1 similarity to hv3: {FHRR.cosine_similarity(hv1, hv3):.3f}")

'''
Yes make three separate binds with a random vector
3:28
To get three new vectors
'''
hv_new = fhrr.random()
hv1b = FHRR.bind(hv1, hv_new)
hv2b = FHRR.bind(hv2, hv_new)
hv3b = FHRR.bind(hv3, hv_new)

#take similarities of each of the new vectors to eachother
print(f"hv1b similarity to hv2b: {FHRR.cosine_similarity(hv1b, hv2b):.3f}")
print(f"hv2b similarity to hv3b: {FHRR.cosine_similarity(hv2b, hv3b):.3f}")
print(f"hv1b similarity to hv3b: {FHRR.cosine_similarity(hv1b, hv3b):.3f}")
