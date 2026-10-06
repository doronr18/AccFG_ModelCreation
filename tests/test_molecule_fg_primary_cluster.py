import sys
from pathlib import Path

import numpy as np

MODULE_DIR = Path(__file__).resolve().parents[1] / 'molecule-fg-data'
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from bernoulli_mixture_clustering import assign_overlapping  # noqa: E402


def test_primary_cluster_is_most_probable():
    resp = np.array([
        [0.2, 0.7, 0.1],   # only cluster 1 qualifies
        [0.35, 0.65, 0.0],  # cluster 1 is most probable; cluster 0 joins as secondary
        [0.1, 0.4, 0.5],   # cluster 2 is most probable; cluster 1 joins as secondary
    ])

    memberships = assign_overlapping(resp, tau=0.3, top_n=2)

    assert memberships == [[1], [1, 0], [2, 1]]
    assert [m[0] for m in memberships] == [int(i) for i in resp.argmax(axis=1)]
