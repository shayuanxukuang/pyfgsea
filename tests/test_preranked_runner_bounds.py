"""Native runner pathway-size enforcement after member deduplication."""

import numpy as np
import pytest

from pyfgsea.wrapper import GseaPrerankedRunner


@pytest.mark.parametrize(
    "pathway,min_size,max_size,deduplicated_size",
    [
        ([0, 0, 1], 3, 5, 2),
        ([0, 1, 2, 3, 4, 4], 1, 4, 5),
    ],
    ids=["below-min-after-dedup", "above-max-after-dedup"],
)
def test_native_runner_rejects_deduplicated_pathways_outside_bounds(
    pathway, min_size, max_size, deduplicated_size
):
    with pytest.raises(
        ValueError,
        match=rf"pathway 0 has deduplicated size {deduplicated_size}, outside",
    ):
        GseaPrerankedRunner([pathway], min_size, max_size)


def test_native_runner_accepts_inclusive_boundaries_and_uses_deduplicated_sizes():
    runner = GseaPrerankedRunner(
        [
            [2, 0, 0, 1],
            [6, 5, 4, 3, 6],
        ],
        3,
        4,
    )
    results = runner.run(
        np.linspace(4.0, -4.0, 12),
        21,
        17,
        1.0,
        0.0,
        score_type="pos",
        precheck_n=101,
        precheck_eps=0.0,
        mode="fast",
        max_levels=100,
    )

    assert len(results) == 2
    assert [result.null_curve_size for result in results] == [3, 4]
