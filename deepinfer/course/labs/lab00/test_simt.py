"""基础设施回归：与学生实现无关，用于锁定简化模型的边界。"""

import unittest

from simt import GPU


class TestMemoryModel(unittest.TestCase):
    def test_shifted_contiguous(self):
        gpu = GPU(64)
        gpu.run(lambda tid, bid, block, grid, dev: dev.load(16 + tid), 1, 32)
        self.assertEqual(gpu.transactions(), 2)
        self.assertEqual(gpu.sectors(), 4)
        self.assertEqual(gpu.stats()["bytes_moved"], 128)
        self.assertEqual(gpu.stats()["sector_bytes"], 128)

    def test_sector_misalignment(self):
        gpu = GPU(64)
        gpu.run(lambda tid, bid, block, grid, dev: dev.load(1 + tid), 1, 32)
        self.assertEqual(gpu.transactions(), 2)
        self.assertEqual(gpu.sectors(), 5)

    def test_stride_eight(self):
        gpu = GPU(256)
        gpu.run(lambda tid, bid, block, grid, dev: dev.load(tid * 8), 1, 32)
        self.assertEqual(gpu.sectors(), 32)
        self.assertEqual(gpu.transactions(), 8)

    def test_shared_broadcast(self):
        gpu = GPU(0, 1)
        gpu.run(lambda tid, bid, block, grid, dev: dev.shared_load(0), 1, 32)
        self.assertEqual(gpu.bank_conflicts(), 0)

    def test_shared_stride(self):
        for stride, expected in ((1, 0), (2, 1), (8, 7), (32, 31), (33, 0)):
            with self.subTest(stride=stride):
                gpu = GPU(0, 32 * stride)
                gpu.run(lambda tid, bid, block, grid, dev: dev.shared_load(tid * stride), 1, 32)
                self.assertEqual(gpu.bank_conflicts(), expected)


if __name__ == "__main__":
    unittest.main()
