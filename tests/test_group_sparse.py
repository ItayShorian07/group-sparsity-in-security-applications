import unittest

import numpy as np
import pandas as pd

from traffic_vae.anomalography import solve_lasso
from traffic_vae.group_sparse import partition, shrink_scores, solve_group_lasso_batch
from traffic_vae.experiment_03 import matched_attacks, metadata_groups, group_scores


class GroupSparseTests(unittest.TestCase):
    def test_identity_solution_matches_exact_block_shrinkage(self):
        targets = np.array([[3., 4., .1, .1], [-1., 2., 0., 0.]])
        groups = [np.array([0, 1]), np.array([2, 3])]
        expected = np.array([shrink_scores(row, [0, 0, 1, 1], .5) for row in targets])
        actual, diagnostics = solve_group_lasso_batch(np.eye(4), targets, groups, .5)
        np.testing.assert_allclose(actual, expected, atol=1e-10)
        self.assertTrue(all(row['converged'] for row in diagnostics))

    def test_singletons_match_lasso_on_nonorthogonal_design(self):
        rng = np.random.default_rng(8)
        design = rng.normal(size=(10, 7))
        targets = rng.normal(size=(4, 10))
        groups = [np.array([i]) for i in range(7)]
        actual, diagnostics = solve_group_lasso_batch(design, targets, groups, .2, tolerance=1e-8)
        expected = np.array([solve_lasso(design, row, .2, tolerance=1e-10)[0] for row in targets])
        np.testing.assert_allclose(actual, expected, atol=1e-6)
        self.assertTrue(all(row['converged'] for row in diagnostics))

    def test_shuffled_coordinates_and_unequal_groups_are_supported(self):
        target = np.array([[3., 1., 4., 2., 0.]])
        groups = [np.array([2, 0]), np.array([4, 1, 3])]
        actual, diagnostics = solve_group_lasso_batch(np.eye(5), target, groups, .3)
        expected = shrink_scores(target[0], [0, 1, 0, 1, 1], .3)
        np.testing.assert_allclose(actual[0], expected, atol=1e-10)
        self.assertTrue(diagnostics[0]['converged'])

    def test_bad_partition_is_rejected(self):
        with self.assertRaises(ValueError):
            partition([[0, 1], [1, 2]], 3)

    def test_score_lasso_is_soft_thresholding(self):
        scores = np.array([.1, .3, 2., 3.])
        result = shrink_scores(scores, np.arange(4), .2)
        np.testing.assert_allclose(result, np.maximum(scores - .2, 0))

    def test_matched_attacks_have_equal_support_energy_and_times(self):
        flows = np.array([(source, target) for source in range(5) for target in range(5) if source != target])
        cases, groups = matched_attacks(flows, 100, 20, 17)
        dispersed, grouped = cases['matched_dispersed'], cases['destination_grouped']
        np.testing.assert_array_equal(np.count_nonzero(dispersed, axis=1), np.count_nonzero(grouped, axis=1))
        self.assertEqual(np.count_nonzero(grouped), 80)
        self.assertEqual(np.linalg.norm(dispersed), np.linalg.norm(grouped))
        for row in grouped[np.any(grouped != 0, axis=1)]:
            self.assertEqual(sum(np.any(row[group]) for group in groups), 1)

    def test_general_group_solution_has_small_primal_dual_gap(self):
        rng = np.random.default_rng(12)
        design, target = rng.normal(size=(7, 12)), rng.normal(size=(1, 7))
        groups = [np.arange(index, index + 3) for index in range(0, 12, 3)]
        penalty = .3
        estimate, _ = solve_group_lasso_batch(design, target, groups, penalty, tolerance=1e-8)
        a = estimate[0]
        residual = target[0] - design @ a
        weights = np.sqrt([len(group) for group in groups])
        primal = .5 * (residual @ residual) + penalty * sum(
            weight * np.linalg.norm(a[group]) for weight, group in zip(weights, groups))
        ratio = max(np.linalg.norm(design[:, group].T @ residual) / (penalty * weight)
                    for group, weight in zip(groups, weights))
        dual_point = residual / max(1, ratio)
        dual = target[0] @ dual_point - .5 * (dual_point @ dual_point)
        self.assertLess(primal - dual, 1e-6)

    def test_group_metadata_does_not_use_labels(self):
        frame = pd.DataFrame({'Destination IP': ['a', 'a', 'b', 'a'],
                              'Timestamp': ['03/07/2017 08:55:01', '03/07/2017 08:55:58',
                                            '03/07/2017 08:55:03', '03/07/2017 08:56:00'],
                              'Label': ['BENIGN', 'DDOS', 'BENIGN', 'DDOS']})
        before = metadata_groups(frame, np.arange(4), 60)
        frame['Label'] = 'CHANGED'
        np.testing.assert_array_equal(before, metadata_groups(frame, np.arange(4), 60))
        np.testing.assert_array_equal(before, [0, 0, 1, 2])

    def test_identity_group_rms_is_shifted_not_new_group_ranking(self):
        scores, ids = np.array([3., 4., 1., 2.]), np.array([0, 0, 1, 1])
        shrunk = shrink_scores(scores, ids, .4)
        np.testing.assert_allclose(group_scores(shrunk, ids), np.maximum(group_scores(scores, ids) - .4, 0))


if __name__ == '__main__':
    unittest.main()
