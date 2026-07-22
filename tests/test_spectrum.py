from collections import Counter

import pandas as pd
import pytest

from selfdgs.spectrum import (
    DGSError,
    all_dgs_configs,
    counters_to_dataframe,
    dataframe_to_dgs,
    dgs_to_dataframe,
    filter_polymorphic_dgs,
    validate_dgs_config,
    validate_dgs_counts,
)


def test_validate_dgs_config_checks_shape_and_sample_size():
    assert validate_dgs_config((1, 2, 1), n_diploids=4) == (1, 2, 1)

    with pytest.raises(DGSError, match="exactly three"):
        validate_dgs_config((1, 2), n_diploids=4)

    with pytest.raises(DGSError, match="sums to"):
        validate_dgs_config((1, 1, 1), n_diploids=4)

    with pytest.raises(DGSError, match="non-integer"):
        validate_dgs_config((1, 1.5, 1))


def test_all_dgs_configs_can_exclude_monomorphic_cells():
    configs = all_dgs_configs(2, polymorphic_only=True)
    assert (2, 0, 0) not in configs
    assert (0, 0, 2) not in configs
    assert (1, 0, 1) in configs
    assert len(configs) == 4


def test_filter_polymorphic_dgs_removes_monomorphic_and_zero_counts():
    observed = Counter({(4, 0, 0): 2, (0, 0, 4): 1, (1, 2, 1): 3, (2, 1, 1): 0})
    assert filter_polymorphic_dgs(observed, n_diploids=4) == Counter({(1, 2, 1): 3})


def test_dataframe_roundtrip_for_tidy_dgs_counts(tmp_path):
    counts = Counter({(1, 2, 1): 3, (2, 1, 1): 4})
    frame = dgs_to_dataframe(counts)

    assert list(frame.columns) == ["n0", "n1", "n2", "count"]
    assert dataframe_to_dgs(frame) == counts


def test_counters_to_dataframe_uses_multiindex():
    frame = counters_to_dataframe(
        [Counter({(1, 2, 1): 3}), Counter({(1, 2, 1): 4, (2, 1, 1): 1})],
        names=["a", "b"],
    )

    assert frame.loc[(1, 2, 1), "a"] == 3
    assert frame.loc[(1, 2, 1), "b"] == 4
    assert frame.loc[(2, 1, 1), "a"] == 0


def test_dataframe_to_dgs_requires_columns():
    with pytest.raises(DGSError, match="missing required columns"):
        dataframe_to_dgs(pd.DataFrame({"n0": [1], "count": [2]}))


def test_validate_dgs_counts_rejects_negative_counts():
    with pytest.raises(DGSError, match="negative"):
        validate_dgs_counts({(1, 2, 1): -1}, n_diploids=4)

    with pytest.raises(DGSError, match="not an integer"):
        validate_dgs_counts({(1, 2, 1): 1.5}, n_diploids=4)
