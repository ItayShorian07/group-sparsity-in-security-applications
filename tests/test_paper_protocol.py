import unittest

import numpy as np

from traffic_vae.anomalography import OnlineAnomalography, solve_lasso
from traffic_vae.paper_experiment import load_config
from traffic_vae.paper_tracking import load_tracking_config
from traffic_vae.paper_synthetic import generate, geometric_routing


class PaperProtocolTests(unittest.TestCase):
    def test_paper_topology_has_expected_dimensions_and_binary_routes(self):
        network = geometric_routing(np.random.default_rng(17))
        self.assertEqual(network['R'].shape, (52, 210))
        self.assertEqual(network['edges'].shape, (52, 2))
        self.assertTrue(np.isin(network['R'], [0, 1]).all())
        self.assertTrue((network['R'].sum(axis=0) >= 1).all())
        self.assertTrue((network['adjacency'] == network['adjacency'].T).all())

    def test_gamma_zero_is_the_paper_linear_endpoint(self):
        config = load_config('experiment_02.json')
        config.update(train_size=30, validation_size=20, calibration_size=20,
                      test_size=200, attack_probability=.02)
        base = generate(config, 42)
        flows, clean, observed = base.observed(0)
        np.testing.assert_allclose(flows['train'], base.linear['train'])
        np.testing.assert_allclose(clean['train'], base.route(base.linear['train'], 'train'))
        np.testing.assert_allclose(observed['train'], clean['train'] + base.noise['train'])
        np.testing.assert_allclose(
            observed['test'], clean['test'] + base.noise['test'] + base.route(base.attacks, 'test')
        )

    def test_routing_alpha_and_observation_probability_follow_paper_notation(self):
        config = load_config('experiment_02.json')
        config.update(train_size=100, validation_size=50, calibration_size=50,
                      test_size=300, attack_probability=.02,
                      routing_alpha=.1, observation_probability=.8)
        base = generate(config, 2026)
        self.assertGreater(len(base.routing_banks['test']), 1)
        self.assertTrue((base.routing_ids['test'] < len(base.routing_banks['test'])).all())
        self.assertTrue((base.routing_banks['test'].sum(axis=1).sum(axis=1) > 0).all())
        self.assertGreater(base.masks['test'].mean(), .7)
        self.assertLess(base.masks['test'].mean(), .9)
        self.assertTrue(base.masks['test'].any(axis=1).all())

    def test_lasso_identity_matches_soft_threshold(self):
        target = np.array([-.8, -.05, .4])
        estimate, diagnostics = solve_lasso(np.eye(3), target, .1)
        expected = np.sign(target) * np.maximum(np.abs(target) - .1, 0)
        np.testing.assert_allclose(estimate, expected, atol=1e-6)
        self.assertTrue(diagnostics['converged'])

    def test_online_tracker_accepts_time_varying_routes_and_masks(self):
        rng = np.random.default_rng(9)
        values = rng.normal(size=(30, 6))
        tracker = OnlineAnomalography(2, .36, .11, routing=np.eye(6),
                                      forgetting_factor=.9, output_after_update=True)
        tracker.initialize(values[:10])
        routes = np.repeat(np.eye(6)[None], 5, axis=0)
        routes[2] = np.roll(np.eye(6), 1, axis=1)
        masks = np.ones((5, 6), dtype=bool)
        masks[:, 0] = False
        first_link_before = tracker.basis[0].copy()
        result = tracker.process(values[10:15], routings=routes, masks=masks)
        self.assertEqual(result.low_rank.shape, (5, 6))
        self.assertEqual(result.sparse.shape, (5, 6))
        self.assertTrue(all(row['observed_links'] == 5 for row in result.history))
        np.testing.assert_allclose(tracker.basis[0], first_link_before)

    def test_tracking_scenarios_use_the_paper_alpha_values(self):
        tracking = load_tracking_config('experiment_02_tracking.json')
        self.assertEqual({item['routing_alpha'] for item in tracking['scenarios']}, {.01, .1})
        self.assertEqual({item['observation_probability'] for item in tracking['scenarios']}, {.8, 1})
        self.assertEqual(tracking['gammas'], [0])


if __name__ == '__main__':
    unittest.main()
