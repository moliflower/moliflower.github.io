"""Reference implementation for the finite teaching pool."""


class BlockPool:
    def __init__(self, capacity):
        if type(capacity) is not int or capacity < 0:
            raise ValueError("invalid capacity")
        self.refs = [0] * capacity

    def available(self):
        return self.refs.count(0)

    def allocate(self, count):
        if type(count) is not int or count < 0:
            raise ValueError("invalid count")
        free = [block for block, refs in enumerate(self.refs) if refs == 0]
        if count > len(free):
            raise MemoryError("pool exhausted")
        chosen = free[:count]
        for block in chosen:
            self.refs[block] = 1
        return chosen

    def _validate(self, block_ids):
        if not isinstance(block_ids, list):
            raise ValueError("IDs must be a list")
        if any(type(block) is not int or not 0 <= block < len(self.refs) or self.refs[block] == 0 for block in block_ids):
            raise ValueError("invalid or free block")
        if len(set(block_ids)) != len(block_ids):
            raise ValueError("duplicate block")

    def retain(self, block_ids):
        self._validate(block_ids)
        for block in block_ids:
            self.refs[block] += 1

    def release(self, block_ids):
        self._validate(block_ids)
        for block in block_ids:
            self.refs[block] -= 1


def locate(block_table, token_index, block_size):
    if not isinstance(block_table, list) or any(type(block) is not int or block < 0 for block in block_table):
        raise ValueError("invalid table")
    if type(block_size) is not int or block_size <= 0 or type(token_index) is not int:
        raise ValueError("invalid index or size")
    if not 0 <= token_index < len(block_table) * block_size:
        raise ValueError("index outside allocated slots")
    logical, offset = divmod(token_index, block_size)
    return block_table[logical], offset
