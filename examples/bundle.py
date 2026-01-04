import os
import sys
import cupy as cp
from vsa_fhrr.algebra import FHRR 



fhrr = FHRR(dim=10000)

hv1 = fhrr.random()
hv2 = fhrr.random()
hv3 = fhrr.random()


centroid_hv = fhrr.centroid(cp.array([hv1, hv2, hv3]))

#similarity comparison

print(f"hv1 similarity to centroid: {FHRR.cosine_similarity(hv1, centroid_hv):.3f}")
print(f"hv2 similarity to centroid: {FHRR.cosine_similarity(hv2, centroid_hv):.3f}")
print(f"hv3 similarity to centroid: {FHRR.cosine_similarity(hv3, centroid_hv):.3f}")
print(f"Random hv similarity to centroid: {FHRR.cosine_similarity(fhrr.random(), centroid_hv):.5f}")

