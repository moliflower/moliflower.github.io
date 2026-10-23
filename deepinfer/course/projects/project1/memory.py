"""Provided scalar-request counter. No cache, hardware concurrency or timing.

Use load_a/load_b/store_c at the global-to-tile boundary. Local tile accesses
are ordinary Python list operations and do not increment these counters.
"""


class MatrixMemory:
    def __init__(self, a, b):
        for name, matrix in (("a", a), ("b", b)):
            if (not isinstance(matrix, list) or not matrix
                    or any(not isinstance(row, list) or not row for row in matrix)
                    or any(len(row) != len(matrix[0]) for row in matrix)):
                raise ValueError(f"{name} must be a nonempty rectangular list of lists")
        if len(a[0]) != len(b):
            raise ValueError("inner dimensions do not match")
        self._a, self._b = a, b
        self.m, self.k, self.n = len(a), len(b), len(b[0]) if b else 0
        self.output = [[0.0] * self.n for _ in range(self.m)]
        self.counts = {"a_reads": 0, "b_reads": 0, "c_writes": 0}

    @staticmethod
    def _check_index(row, col, rows, cols):
        # Python accepts negative indices; a tile memory model must not.
        if type(row) is not int or type(col) is not int or not (0 <= row < rows and 0 <= col < cols):
            raise IndexError("matrix address is outside the valid region")

    def load_a(self, i, k):
        self._check_index(i, k, self.m, self.k)
        value = float(self._a[i][k])
        self.counts["a_reads"] += 1
        return value

    def load_b(self, k, j):
        self._check_index(k, j, self.k, self.n)
        value = float(self._b[k][j])
        self.counts["b_reads"] += 1
        return value

    def store_c(self, i, j, value):
        self._check_index(i, j, self.m, self.n)
        self.counts["c_writes"] += 1
        self.output[i][j] = value

    def stats(self):
        return {**self.counts, "requested_bytes": 4 * sum(self.counts.values())}
