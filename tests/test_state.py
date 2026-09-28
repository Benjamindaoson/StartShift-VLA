import numpy as np

from startshift.data.state import MeanStdStats, augment_normalized_state


def test_state_representation_preserves_absolute_initial_and_relative():
    current = np.array([1.0, 2.0], dtype=np.float32)
    initial = np.array([0.25, 1.5], dtype=np.float32)
    out = augment_normalized_state(current, initial)
    np.testing.assert_allclose(out, [1.0, 2.0, 0.25, 1.5, 0.75, 0.5])


def test_mean_std_round_trip():
    stats = MeanStdStats(
        mean=np.array([2.0, 4.0], dtype=np.float32),
        std=np.array([2.0, 4.0], dtype=np.float32),
    )
    x = np.array([4.0, 8.0], dtype=np.float32)
    np.testing.assert_allclose(stats.denormalize(stats.normalize(x)), x)
