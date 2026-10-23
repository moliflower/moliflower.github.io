"""Lab4 finite shared block pool. See README.md for the full contract."""


class BlockPool:
    def __init__(self, capacity):
        if type(capacity) is not int or capacity < 0:
            raise ValueError("capacity must be a nonnegative integer")
        self.refs = [0] * capacity

    def available(self):
        return self.refs.count(0)

    def allocate(self, count):
        """Q1: lowest free IDs, refs=1; invalid -> ValueError, insufficient -> MemoryError. Atomic."""
        raise NotImplementedError("Lab4 Q1")

    def retain(self, block_ids):
        """Q2: increment each unique occupied ID. Invalid IDs/duplicates -> ValueError. Atomic."""
        raise NotImplementedError("Lab4 Q2")

    def release(self, block_ids):
        """Q3: decrement each unique occupied ID; zero becomes free. Same validation as retain."""
        raise NotImplementedError("Lab4 Q3")


def locate(block_table, token_index, block_size):
    """Q4: (physical block ID, offset); integer inputs exclude bool. See README."""
    raise NotImplementedError("Lab4 Q4")
