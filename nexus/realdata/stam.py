import numpy as np
from nexus.realdata.creamer import identified_stims, label_index

def stam_sums(recordings, labels, window, baseline=0) -> tuple:
    """Calculate the sum of areas for each pair of neurons."""
    n = len(labels)
    total = np.zeros((n, n))
    count = np.zeros((n, n))
    for r in recordings:
        x = r["neural"]
        ids = r["ids"]
        cells, frames = identified_stims(r)
        for c, f in zip(cells, frames):
            if not ids[c] or f + window > x.shape[0] or (baseline > 0 and f - baseline < 0):
                continue
            seg = x[f:f + window, :]
            for j in range(x.shape[1]):
                if not ids[j] or np.all(np.isnan(seg[:, j])):
                    continue
                area = np.nanmean(seg[:, j]) * window
                if baseline > 0:
                    pre = x[f - baseline:f, j]
                    if np.all(np.isnan(pre)):
                        continue
                    area = area - np.nanmean(pre) * window
                a = labels[ids[j]]
                b = labels[ids[c]]
                total[a, b] += area
                count[a, b] += 1
    return total, count

def stam(recordings, labels, window, baseline=0, min_events=1) -> np.ndarray:
    """Calculate the average area for each pair of neurons with a minimum number of events."""
    total, count = stam_sums(recordings, labels, window, baseline)
    result = np.full_like(total, np.nan)
    result[count >= min_events] = total[count >= min_events] / count[count >= min_events]
    return result

def matrix_correlation(a, b, off_diagonal=True) -> tuple:
    """Calculate the correlation coefficient and number of finite elements."""
    use = np.isfinite(a) & np.isfinite(b)
    if off_diagonal:
        np.fill_diagonal(use, False)
    return (float(np.corrcoef(a[use], b[use])[0, 1]), int(use.sum()))

def train_test_correlation(recordings, n_train, window, seed, baseline=0, min_events=1, off_diagonal=True) -> tuple:
    """Calculate the correlation between training and test sets."""
    labels = label_index(recordings)
    order = np.random.default_rng(seed).permutation(len(recordings))
    train = [recordings[int(i)] for i in order[:n_train]]
    test = [recordings[int(i)] for i in order[n_train:]]
    return matrix_correlation(stam(train, labels, window, baseline, min_events), stam(test, labels, window, baseline, min_events), off_diagonal)
