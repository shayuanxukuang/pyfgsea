import math

import numpy as np
import pytest
from scipy.special import polygamma

from pyfgsea.wrapper import build_tail_curve, query_tail_curve


def _build_curve(*, score_type="std", sign=1, eps=0.0):
    with pytest.warns(DeprecationWarning, match="legacy empirical"):
        return build_tail_curve(
            np.linspace(5.0, -5.0, 24),
            4,
            31,
            17,
            1.0,
            eps,
            score_type=score_type,
            sign=sign,
        )


def test_legacy_tail_curve_is_explicitly_approximate_and_warns_on_query():
    curve = _build_curve(score_type="std", sign=-1)

    assert curve.approximate is True
    assert curve.algorithm_revision == "legacy-empirical-tail-v1"
    assert curve.score_type == "std"
    assert curve.sign == -1
    assert "legacy empirical tail curve" in curve.termination_reason

    observed = -0.4
    population = np.asarray(curve.populations[-1])
    count = int(np.count_nonzero(population >= -observed))
    expected_pvalue = (count + 1) / (curve.sample_size + 1)
    expected_log2err = math.sqrt(
        float(polygamma(1, count + 1) - polygamma(1, curve.sample_size + 1))
    ) / math.log(2.0)

    with pytest.warns(DeprecationWarning, match="aligned multilevel"):
        pvalue, log2err = query_tail_curve(curve, observed)

    assert pvalue == pytest.approx(expected_pvalue)
    assert log2err == pytest.approx(expected_log2err)
    assert math.isfinite(log2err)


@pytest.mark.parametrize(
    ("score_type", "sign", "message"),
    [("pos", -1, "score_type mismatch"), ("std", 1, "sign mismatch")],
)
def test_legacy_tail_curve_rejects_query_contract_mismatch(
    score_type, sign, message
):
    curve = _build_curve(score_type="std", sign=-1)

    with pytest.warns(DeprecationWarning):
        with pytest.raises(ValueError, match=message):
            query_tail_curve(curve, -0.4, score_type=score_type, sign=sign)
