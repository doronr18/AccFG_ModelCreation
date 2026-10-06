import sys
from pathlib import Path

import numpy as np

MODULE_DIR = Path(__file__).resolve().parents[1] / 'molecule-fg-data'
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

import bernoulli_mixture_clustering as bmc  # noqa: E402


def test_log_joint_matches_per_component_loop():
    rng = np.random.default_rng(0)
    X = (rng.random((50, 20)) < 0.3).astype(np.float64)
    means = rng.uniform(0.0, 1.0, size=(4, 20))
    means[0, :3] = 0.0  # exact 0 and 1 exercise the smoothing
    means[1, :3] = 1.0
    weights = np.array([0.4, 0.3, 0.2, 0.1])

    expected = np.zeros((50, 4))
    for k in range(4):
        expected[:, k] = np.log(weights[k] + 1e-12) + (
            X * np.log(means[k] + 1e-12) + (1 - X) * np.log(1 - means[k] + 1e-12)
        ).sum(axis=1)

    np.testing.assert_allclose(bmc._log_joint(X, means, weights), expected, rtol=1e-10, atol=1e-8)
