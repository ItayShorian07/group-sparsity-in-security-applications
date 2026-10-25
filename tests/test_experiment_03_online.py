import logging
from pathlib import Path
import tempfile
import unittest

import numpy as np

from traffic_vae.anomalography import OnlineAnomalography
from traffic_vae.experiment_03_online import independent_stream, restore_tracker


class Experiment3OnlineTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(17)
        self.routing = self.rng.normal(size=(5, 7))
        self.values = self.rng.normal(size=(40, 5))
        self.tracker = OnlineAnomalography(2, .36, .11, routing=self.routing, output_after_update=True)
        self.tracker.initialize(self.values[:12])
        self.tracker.process(self.values[12:20])
        self.config = dict(da_rank=2, da_ridge_penalty=.36, da_sparse_penalty=.11,
                           lasso_max_iterations=3000, lasso_tolerance=1e-7, da_forgetting_factor=1.)

    def test_saved_state_restores_identical_adaptive_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'state.npz'
            np.savez(path, basis=self.tracker.basis, gram=self.tracker.gram, cross=self.tracker.cross)
            restored = restore_tracker(path, self.routing, self.config)
            expected = self.tracker.copy().process(self.values[20:])
            actual = restored.process(self.values[20:])
        np.testing.assert_array_equal(expected.sparse, actual.sparse)
        np.testing.assert_array_equal(expected.low_rank, actual.low_rank)

    def test_streams_are_independent_and_chunking_does_not_reset_updates(self):
        before = [value.copy() for value in (self.tracker.basis, self.tracker.gram, self.tracker.cross)]
        logger = logging.getLogger('test_online')
        expected = self.tracker.copy().process(self.values[20:])
        first = independent_stream(self.tracker, self.values[20:], logger, 'first', chunk_size=3)
        independent_stream(self.tracker, self.values[20:] + 10, logger, 'other', chunk_size=7)
        second = independent_stream(self.tracker, self.values[20:], logger, 'second', chunk_size=11)
        np.testing.assert_array_equal(first.sparse, expected.sparse)
        np.testing.assert_array_equal(first.sparse, second.sparse)
        self.assertEqual([row['time_index'] for row in first.history], list(range(20)))
        for old, current in zip(before, (self.tracker.basis, self.tracker.gram, self.tracker.cross)):
            np.testing.assert_array_equal(old, current)

    def test_invalid_saved_state_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'state.npz'
            np.savez(path, basis=np.zeros((5, 1)), gram=self.tracker.gram, cross=self.tracker.cross)
            with self.assertRaises(ValueError):
                restore_tracker(path, self.routing, self.config)


if __name__ == '__main__':
    unittest.main()
