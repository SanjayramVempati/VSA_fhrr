"""Analyze encoded hypervectors saved by `fpe_time_encoding.py`.
Computes pairwise similarity samples, intra/inter-class means, and saves a histogram and PCA scatter.
"""
import os
import sys
import numpy as np
import random
import matplotlib.pyplot as plt


def load_encodings(path):
    r = np.load(path)
    return r['encodings'], r['labels']


def cos_sim(a, b):
    return float(np.real(np.vdot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b)))


def sample_pairwise_sims(E, n_samples=2000):
    N = E.shape[0]
    sims = []
    for _ in range(n_samples):
        i = random.randrange(N)
        j = random.randrange(N)
        sims.append(cos_sim(E[i], E[j]))
    return np.array(sims)


def intra_inter_stats(E, labels, max_per_class=200):
    uniq = np.unique(labels)
    stats = {}
    for u in uniq:
        idx = np.where(labels == u)[0]
        if len(idx) < 2:
            stats[u] = {'intra_mean': None, 'count': len(idx)}
            continue
        idxs = idx if len(idx) <= max_per_class else np.random.choice(idx, max_per_class, replace=False)
        sims = []
        for i in range(len(idxs)):
            for j in range(i + 1, len(idxs)):
                sims.append(cos_sim(E[idxs[i]], E[idxs[j]]))
        stats[u] = {'intra_mean': float(np.mean(sims)), 'intra_std': float(np.std(sims)), 'count': len(idx)}

    # inter-class pair mean (sampled)
    inter_sims = []
    uniq_list = list(uniq)
    for _ in range(2000):
        a = random.choice(uniq_list)
        b = random.choice(uniq_list)
        if a == b:
            continue
        ia = random.choice(np.where(labels == a)[0])
        ib = random.choice(np.where(labels == b)[0])
        inter_sims.append(cos_sim(E[ia], E[ib]))
    stats['_inter_mean'] = float(np.mean(inter_sims))
    stats['_inter_std'] = float(np.std(inter_sims))
    return stats


def pca_2d(E):
    # center
    X = E - E.mean(axis=0)
    # SVD
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    coords = U[:, :2] * S[:2]
    return coords


def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    # allow overriding the encodings path via first CLI arg
    if len(sys.argv) > 1:
        enc_path = sys.argv[1]
    else:
        enc_path = os.path.join(project_root, 'recordings', 'encoded_fpe.npz')
    if not os.path.exists(enc_path):
        print('Encodings file not found:', enc_path)
        sys.exit(1)

    E, labels = load_encodings(enc_path)
    print('Loaded encodings:', E.shape, 'labels:', labels.shape)

    sims = sample_pairwise_sims(E, n_samples=4000)
    print(f'Pairwise sample sims: mean={sims.mean():.4f}, std={sims.std():.4f}, min={sims.min():.4f}, max={sims.max():.4f}')

    stats = intra_inter_stats(E, labels)
    print('Per-class intra stats:')
    for k, v in stats.items():
        # skip special keys like '_inter_mean'
        if isinstance(k, str) and k.startswith('_'):
            continue
        print(' ', k, v)
    print('Inter-class mean/std:', stats.get('_inter_mean'), stats.get('_inter_std'))

    out_dir = os.path.join(project_root, 'recordings')
    # histogram
    plt.figure(figsize=(6, 4))
    plt.hist(sims, bins=60)
    plt.title('Sampled pairwise cosine similarities')
    plt.xlabel('cosine similarity')
    plt.ylabel('count')
    hist_path = os.path.join(out_dir, 'similarity_hist.png')
    plt.tight_layout()
    plt.savefig(hist_path)
    print('Saved histogram to', hist_path)

    # PCA scatter
    coords = pca_2d(E)
    plt.figure(figsize=(6, 6))
    uniq = np.unique(labels)
    colors = plt.cm.get_cmap('tab10', len(uniq))
    for i, u in enumerate(uniq):
        idx = np.where(labels == u)[0]
        plt.scatter(coords[idx, 0], coords[idx, 1], s=8, color=colors(i), label=str(u))
    plt.legend(markerscale=2)
    plt.title('PCA (2D) of encodings')
    pca_path = os.path.join(out_dir, 'pca_encodings.png')
    plt.tight_layout()
    plt.savefig(pca_path)
    print('Saved PCA scatter to', pca_path)


if __name__ == '__main__':
    main()
