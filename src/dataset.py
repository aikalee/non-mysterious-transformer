import numpy as np

def make_concentric_rings(n_per_class=1000, radii=(0.5, 1.0, 1.5), width=0.07, seed=0):
    rng = np.random.default_rng(seed)
    Xs, ys = [], []
    for cls, r0 in enumerate(radii):
        r = r0 + rng.normal(0.0, width, size=n_per_class)
        th = rng.uniform(0.0, 2*np.pi, size=n_per_class)
        x = r * np.cos(th)
        y = r * np.sin(th)
        Xs.append(np.stack([x, y], axis=1))
        ys.append(np.full(n_per_class, cls, dtype=np.int64))
    X = np.concatenate(Xs, axis=0).astype(np.float32)
    y = np.concatenate(ys, axis=0)
    idx = rng.permutation(len(y))
    return X[idx], y[idx]