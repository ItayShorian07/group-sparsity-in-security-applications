import copy
import json
from pathlib import Path
import unittest

import numpy as np

from traffic_vae.high_rank_sensitivity import fourier_features, replace_nonlinear
from traffic_vae.paper_experiment import load_config
from traffic_vae.paper_synthetic import generate


class HighRankSensitivityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.settings = json.loads(Path('high-rank-sensitivity.json').read_text())
        config = load_config('experiment-02.json')
        config.update(train_size=4000, validation_size=30, calibration_size=30, test_size=200)
        cls.original = generate(config, 17)

    def test_high_rank_with_unchanged_linear_control_and_attack_budget(self):
        base = copy.deepcopy(self.original)
        audit = replace_nonlinear(base, 17, self.settings)
        self.assertGreaterEqual(audit['rank_95'], 12)
        self.assertLess(audit['top5_energy_fraction'], .5)
        self.assertEqual(audit['intrinsic_latent_dimension'], 2)
        before = self.original.observed(0)[2]
        after = base.observed(0)[2]
        for split in before:
            np.testing.assert_array_equal(before[split], after[split])
        np.testing.assert_array_equal(base.attacks, self.original.attacks)
        np.testing.assert_array_equal(base.noise['test'], self.original.noise['test'])
        rms = lambda value: np.sqrt(np.mean(value**2))
        self.assertAlmostEqual(rms(base.route(base.nonlinear['train'], 'train')),
                               rms(base.route(base.linear['train'], 'train')), places=12)

    def test_heldout_data_and_attack_labels_do_not_choose_the_generator(self):
        first, second = copy.deepcopy(self.original), copy.deepcopy(self.original)
        second.latent['test'] *= 4
        second.latent['calibration'] *= -3
        second.attacks[:] = 10
        replace_nonlinear(first, 17, self.settings)
        replace_nonlinear(second, 17, self.settings)
        np.testing.assert_array_equal(first.nonlinear['train'], second.nonlinear['train'])
        for key in ('H_od_loadings', 'H_link_directions', 'H_training_feature_center'):
            np.testing.assert_array_equal(first.maps[key], second.maps[key])
        self.assertEqual(first.nonlinear_scale, second.nonlinear_scale)

    def test_routed_features_really_match_the_declared_orthonormal_directions(self):
        base = copy.deepcopy(self.original)
        replace_nonlinear(base, 17, self.settings)
        directions = base.maps['H_link_directions']
        np.testing.assert_allclose(directions.T @ directions, np.eye(16), atol=1e-12)
        np.testing.assert_allclose(base.network['R'] @ base.maps['H_od_loadings'], directions, atol=1e-12)

    def test_fourier_modes_reject_hidden_rank_deficiencies(self):
        for modes in ([[0, 0]], [[1, 0], [1, 0]], [[1, 0], [-1, 0]], [[.5, 0]]):
            with self.assertRaises(ValueError):
                fourier_features(np.zeros((3, 2)), modes)
        actual = fourier_features(np.array([[.1, -.3]]), self.settings['fourier_modes'])
        self.assertEqual(actual.shape, (1, 16))
        np.testing.assert_array_equal(actual, fourier_features(np.array([[.1, -.3]]), self.settings['fourier_modes']))


if __name__ == '__main__':
    unittest.main()
