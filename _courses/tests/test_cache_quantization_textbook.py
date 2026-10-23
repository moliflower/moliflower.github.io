import importlib.util
from pathlib import Path
import sys
import unittest


sys.dont_write_bytecode = True
COURSE = Path(__file__).resolve().parents[1] / "deepinfer" / "course"


def reference(name, relative_path):
    specification = importlib.util.spec_from_file_location(name, COURSE / relative_path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


cache = reference("textbook_cache", "labs/lab03/lab03_solution.py")
costs = reference("textbook_costs", "hw/hw01/hw01_solution.py")
pages = reference("textbook_pages", "hw/hw02/hw02_solution.py")


class CacheAndQuantizationTextbook(unittest.TestCase):
    def test_kv_axis_derivation_and_batch_scaling(self):
        per_layer = 2 * 8 * 128 * 2
        self.assertEqual(per_layer, 4096)
        self.assertEqual(costs.kv_cache_bytes(32, 8, 128, 1), 131072)
        self.assertEqual(costs.kv_cache_bytes(32, 8, 128, 2048), 256 * 2**20)
        self.assertEqual(costs.kv_cache_bytes(32, 8, 128, 2048, 32), 8 * 2**30)
        self.assertEqual(costs.kv_cache_bytes(32, 8, 128, 8192), 2**30)
        self.assertEqual([costs.kv_cache_bytes(2, 1, 3, length) for length in [0, 1, 4]], [0, 24, 96])

    def test_budget_exact_boundary_and_maximality(self):
        budget = 512 * 2**20
        self.assertEqual(costs.max_kv_tokens(budget, 32, 8, 128), 4096)
        self.assertEqual(costs.max_kv_tokens(budget - 1, 32, 8, 128), 4095)
        self.assertLessEqual(4096 * 131072, budget)
        self.assertGreater(4097 * 131072, budget)

    def test_page_table_and_large_boundary(self):
        self.assertEqual(pages.num_pages(5000, 64), 79)
        self.assertEqual(pages.page_waste(5000, 64), 56)
        self.assertEqual([(pages.num_pages(length, 64), pages.page_waste(length, 64)) for length in [0, 64, 65]], [(0, 0), (1, 0), (2, 63)])
        self.assertEqual([pages.page_waste(10, size) for size in [4, 16, 128]], [2, 6, 118])
        self.assertEqual([70 // size * size for size in [16, 64, 128]], [64, 64, 0])

    def test_ragged_lengths_are_rounded_individually(self):
        lengths = [1, 16, 17]
        self.assertEqual([pages.num_pages(length, 16) for length in lengths], [1, 1, 2])
        self.assertEqual(sum(pages.page_waste(length, 16) for length in lengths), 30)
        self.assertEqual(pages.paged_batch_bytes(lengths, 16, 1, 1, 1), 256)
        self.assertEqual(pages.paged_kv_bytes(1, 1, 1, sum(lengths), 16), 192)

    def test_physical_mapping_and_byte_offset(self):
        block_table = [7, 2]
        logical, offset = divmod(3, 2)
        physical = block_table[logical]
        self.assertEqual((physical, offset), (2, 1))
        self.assertEqual(physical * 24 + offset * 12 + 1 * 6 + 2 * 2, 70)

    def test_append_boundary_table(self):
        paged = cache.PagedKVCache(2)
        observed = []
        for position in range(5):
            before = len(paged)
            returned = paged.append([position], [-position])
            self.assertEqual(returned, position)
            observed.append((before, before % 2, len(paged), paged.allocated_slots(), paged.waste()))
        self.assertEqual(observed, [(0, 0, 1, 2, 1), (1, 1, 2, 2, 0), (2, 0, 3, 4, 1), (3, 1, 4, 4, 0), (4, 0, 5, 6, 1)])

    def test_gather_after_noncontiguous_remapping(self):
        paged = cache.PagedKVCache(2)
        for position in range(5):
            paged.append([position], [-position])
        mapping = {0: 9, 1: 2, 2: 7}
        paged.blocks = {mapping[block]: content for block, content in paged.blocks.items()}
        paged.block_table = [mapping[block] for block in paged.block_table]
        self.assertEqual(paged.get(1, 5), ([[1.0], [2.0], [3.0], [4.0]], [[-1.0], [-2.0], [-3.0], [-4.0]]))
        self.assertEqual(paged.get(5, 5), ([], []))
        with self.assertRaises(ValueError):
            paged.get(5, 6)

    def test_input_output_copy_and_reset(self):
        contiguous = cache.KVCache(2)
        key, value = [1, 2], [3, 4]
        contiguous.append(key, value)
        key[0] = 99
        returned_key, returned_value = contiguous.get(0, 1)
        returned_key[0][0] = 88
        returned_value[0][0] = 77
        self.assertEqual(contiguous.get(0, 1), ([[1.0, 2.0]], [[3.0, 4.0]]))
        contiguous.free()
        self.assertEqual(contiguous.get(0, 0), ([], []))
        self.assertEqual(contiguous.append([5, 6], [7, 8]), 0)

    def test_mla_dtype_comparison(self):
        self.assertEqual(costs.mla_kv_cache_bytes(32, 512, 64, 2048, 1, 1), 36 * 2**20)
        self.assertEqual(costs.mla_kv_cache_bytes(32, 512, 64, 2048, 1, 2), 72 * 2**20)
        self.assertAlmostEqual(256 / 72, 3.5555555555555554)

    def test_uniform_grid_nonzero_error_and_zeros(self):
        codes, scale = costs.quantize_per_tensor([-1, 0, 1], 3)
        self.assertEqual(codes, [-3, 0, 3])
        self.assertEqual(scale, 1 / 3)
        self.assertEqual(costs.dequantize_per_tensor(codes, scale), [-1.0, 0.0, 1.0])
        self.assertAlmostEqual(costs.quant_error([0, 0.2, 1], 3), 2 / 15)
        self.assertEqual(costs.quantize_per_tensor([], 8), ([], 1.0))
        self.assertEqual(costs.quantize_per_tensor([0, 0], 8), ([0, 0], 1.0))
        with self.assertRaises(ValueError):
            costs.quantize_per_tensor([], True)

    def test_halfway_ties_and_clipping_counterexample(self):
        original = [0.5, 1.5, 2.5, 3]
        codes, scale = costs.quantize_per_tensor(original, 3)
        self.assertEqual((codes, scale), ([0, 2, 2, 3], 1.0))
        self.assertEqual(costs.quant_error(original, 3), 0.5)
        self.assertEqual([round(value) for value in [-0.5, -1.5, -2.5]], [0, -2, -2])
        clipped = max(-3, min(3, round(10 / scale)))
        self.assertEqual(10 - clipped * scale, 7)
        self.assertGreater(abs(10 - clipped * scale), scale / 2)

    def test_outlier_isolation_in_three_element_groups(self):
        original = [0.1, 0.2, 0.3, 100]
        self.assertEqual(costs.quantize_per_tensor(original, 8)[0], [0, 0, 0, 127])
        self.assertAlmostEqual(costs.quant_error(original, 8), 0.3)
        codes, scale = costs.quantize_per_tensor(original[:3], 8)
        self.assertEqual(codes, [42, 85, 127])
        reconstructed = costs.dequantize_per_tensor(codes, scale) + costs.dequantize_per_tensor(*costs.quantize_per_tensor(original[3:], 8))
        grouped_error = max(abs(before - after) for before, after in zip(original, reconstructed))
        self.assertAlmostEqual(grouped_error, 0.1 / 127)

    def test_grouping_can_leave_global_max_error_unchanged(self):
        original = [-1, 0, 1, 100]
        rebuilt = []
        for start in [0, 2]:
            rebuilt.extend(costs.dequantize_per_tensor(*costs.quantize_per_tensor(original[start:start + 2], 8)))
        global_error = costs.quant_error(original, 8)
        grouped_error = max(abs(before - after) for before, after in zip(original, rebuilt))
        self.assertEqual(rebuilt[0], -1)
        self.assertAlmostEqual(grouped_error, global_error)
        self.assertAlmostEqual(global_error, 27 / 127)

    def test_metadata_and_payload_accounting(self):
        self.assertEqual(32 * 4 // 8 + 1, 17)
        self.assertAlmostEqual(64 / 17, 3.764705882352941)
        self.assertEqual(costs.kv_cache_bytes(32, 8, 128, 4096), 512 * 2**20)
        self.assertEqual(costs.kv_cache_bytes(32, 8, 128, 4096, bytes_per_elem=1), 256 * 2**20)
        self.assertEqual(costs.kv_cache_bytes(32, 8, 128, 4096) // 4, 128 * 2**20)

    def test_smaller_half_grid_bound_does_not_guarantee_smaller_error(self):
        original = [1, 2, 3]
        self.assertEqual(costs.quant_error(original, 3), 0)
        codes, scale = costs.quantize_per_tensor(original[:2], 3)
        self.assertEqual(codes, [2, 3])
        rebuilt = costs.dequantize_per_tensor(codes, scale)
        self.assertAlmostEqual(abs(rebuilt[0] - 1), 1 / 3)
        self.assertLess(scale / 2, 1 / 2)

    def test_roofline_tie_labels_preserve_interfaces(self):
        self.assertEqual(costs.roofline_time(1e9, 1e11, 100, 10), (0.01, "memory"))
        self.assertEqual(pages.bound(100, 10, 100), "compute")
        self.assertEqual(pages.bound(64, 100, 1000), "memory")
        self.assertEqual(pages.bound(200, 100, 1000), "compute")

    def test_attention_ledger_and_quiz_anchors(self):
        self.assertEqual(pages.attention_flops(1, 8, 128, 128, 64, False), 33554432)
        self.assertEqual(pages.attention_bytes(1, 8, 128, 128, 64), 1572864)
        self.assertEqual(pages.attention_bytes(1, 8, 128, 128, 64, materialized=False), 524288)
        self.assertEqual(pages.attention_flops(2, 8, 512, 512, 64), 537919488)
        self.assertEqual(pages.attention_flops(2, 8, 512, 512, 64, False), 1073741824)
        self.assertEqual(pages.attention_bytes(1, 8, 512, 512, 64), 18874368)
        self.assertEqual(pages.attention_bytes(1, 8, 512, 512, 64, materialized=False), 2097152)
        self.assertEqual(costs.kv_cache_bytes(32, 8, 128, 4096, 4, 1), 2**30)

    def test_growth_and_sharing_baselines(self):
        self.assertEqual(pages.kv_cache_growth(32, 8, 128, 100), 13107200)
        self.assertEqual(pages.paged_kv_bytes(32, 8, 128, 1000, 16), 126 * 2**20)
        self.assertEqual(pages.prefix_cache_savings(32, 8, 128, 512, 3), 192 * 2**20)
        internally_shared_savings = (3 - 1) * costs.kv_cache_bytes(32, 8, 128, 512)
        self.assertEqual(internally_shared_savings, 128 * 2**20)


if __name__ == "__main__":
    unittest.main()
