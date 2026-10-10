"""Maximum-weight one-to-one assignment, including dependency-free CPU fallback."""
import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None


def maximum_assignment(weights, native=True):
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim != 2 or not np.isfinite(weights).all():
        raise ValueError("Assignment weights must be a finite matrix")
    if 0 in weights.shape:
        return []
    if native and linear_sum_assignment is not None:
        rows, columns = linear_sum_assignment(-weights)
        return list(zip(rows.tolist(), columns.tolist(), strict=True))
    transposed = weights.shape[0] > weights.shape[1]
    cost = -weights.T if transposed else -weights
    n, m = cost.shape
    # Shortest augmenting paths with row/column dual potentials. n <= m.
    u, v = np.zeros(n + 1), np.zeros(m + 1)
    owner, previous = np.zeros(m + 1, dtype=int), np.zeros(m + 1, dtype=int)
    for row in range(1, n + 1):
        owner[0], column = row, 0
        distance, used = np.full(m + 1, np.inf), np.zeros(m + 1, dtype=bool)
        while True:
            used[column] = True
            active = owner[column]
            delta, next_column = np.inf, 0
            for candidate in range(1, m + 1):
                if used[candidate]:
                    continue
                reduced = cost[active - 1, candidate - 1] - u[active] - v[candidate]
                if reduced < distance[candidate]:
                    distance[candidate], previous[candidate] = reduced, column
                if distance[candidate] < delta:
                    delta, next_column = distance[candidate], candidate
            for candidate in range(m + 1):
                if used[candidate]:
                    u[owner[candidate]] += delta
                    v[candidate] -= delta
                else:
                    distance[candidate] -= delta
            column = next_column
            if owner[column] == 0:
                break
        while column:
            parent = previous[column]
            owner[column] = owner[parent]
            column = parent
    pairs = [(int(owner[column]) - 1, column - 1) for column in range(1, m + 1) if owner[column]]
    return sorted((column, row) for row, column in pairs) if transposed else sorted(pairs)
