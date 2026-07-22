import math

import pytest

from selfdgs.model import (
    PRACTICAL_MAX_DIPLOIDS,
    dgs_probabilities,
    expected_dgs_branch_lengths,
    fast_class_branch_lengths,
    fast_outcome_probability,
    fast_state_from_outcome,
    folded_dgs_probabilities,
    genotype_config,
    inbreeding_coefficient,
    make_branch_length_recursion,
)
from selfdgs.model import probabilities as probability_module
from selfdgs.spectrum import all_dgs_configs, fold_dgs_config


def test_genotype_config_maps_haplotypes_to_dgs_cell():
    assert genotype_config({0, 1, 3}, n_diploids=2) == (0, 1, 1)
    assert genotype_config({0, 2}, n_diploids=2) == (0, 2, 0)
    assert genotype_config(set(), n_diploids=2) == (2, 0, 0)


def test_fast_state_from_outcome_collapses_homologs():
    assert fast_state_from_outcome((1, 0), n_diploids=2) == (
        frozenset({0, 1}),
        frozenset({2}),
        frozenset({3}),
    )


def test_fast_outcome_probability_multiplies_collapse_probabilities():
    assert fast_outcome_probability((1, 0, 1), F=0.25) == pytest.approx(0.25 * 0.75 * 0.25)


def test_combinatorial_fast_classes_match_articulated_recursion():
    for n_diploids in range(1, 4):
        recursion = make_branch_length_recursion(n_diploids=n_diploids)
        for collapsed in range(n_diploids + 1):
            outcome = (1,) * collapsed + (0,) * (n_diploids - collapsed)
            state = fast_state_from_outcome(outcome, n_diploids=n_diploids)

            assert fast_class_branch_lengths(
                n_diploids=n_diploids,
                collapsed_diploids=collapsed,
            ) == pytest.approx(recursion(state))


@pytest.mark.parametrize("s", [0.0, 0.37, 1.0])
def test_combinatorial_model_matches_articulated_recursion(s):
    combinatorial = expected_dgs_branch_lengths(
        s=s,
        N=7,
        n_diploids=2,
        use_generation_time_scale=True,
        method="combinatorial",
    )
    recursive = expected_dgs_branch_lengths(
        s=s,
        N=7,
        n_diploids=2,
        use_generation_time_scale=True,
        method="recursive",
    )

    assert combinatorial == pytest.approx(recursive)


def test_inbreeding_coefficient_matches_selfing_equilibrium_formula():
    assert inbreeding_coefficient(0.0) == 0.0
    assert inbreeding_coefficient(0.5) == pytest.approx(1 / 3)
    assert inbreeding_coefficient(1.0) == 1.0


def test_probabilities_sum_to_one_and_include_all_polymorphic_cells():
    probabilities = dgs_probabilities(s=0.5, n_diploids=4)

    assert sum(probabilities.values()) == pytest.approx(1.0)
    assert set(probabilities) == set(all_dgs_configs(4, polymorphic_only=True))
    assert all(value >= 0 for value in probabilities.values())


def test_probabilities_handle_boundary_selfing_rates():
    for s in [0.0, 0.999, 1.0]:
        probabilities = dgs_probabilities(s=s, n_diploids=4)
        assert sum(probabilities.values()) == pytest.approx(1.0)
        assert all(math.isfinite(value) for value in probabilities.values())


def test_fold_dgs_config_canonicalizes_allele_orientation():
    assert fold_dgs_config((3, 1, 0), n_diploids=4) == (0, 1, 3)
    assert fold_dgs_config((0, 1, 3), n_diploids=4) == (0, 1, 3)
    assert fold_dgs_config((1, 2, 1), n_diploids=4) == (1, 2, 1)


def test_folded_probabilities_sum_unfolded_orientations():
    unfolded = dgs_probabilities(s=0.5, n_diploids=2)
    folded = folded_dgs_probabilities(s=0.5, n_diploids=2)

    assert sum(folded.values()) == pytest.approx(1.0)
    assert folded[(0, 1, 1)] == pytest.approx(
        unfolded[(0, 1, 1)] + unfolded[(1, 1, 0)]
    )
    assert folded[(0, 2, 0)] == pytest.approx(unfolded[(0, 2, 0)])
    assert folded[(1, 0, 1)] == pytest.approx(unfolded[(1, 0, 1)])


def test_invalid_model_inputs_raise_value_error():
    with pytest.raises(ValueError, match="s must be"):
        dgs_probabilities(s=-0.01, n_diploids=4)

    with pytest.raises(ValueError, match="s must be"):
        expected_dgs_branch_lengths(s=1.01, n_diploids=4)

    with pytest.raises(ValueError, match="N must be positive"):
        expected_dgs_branch_lengths(s=0.5, N=0, n_diploids=4)

    with pytest.raises(ValueError, match="N must be positive"):
        expected_dgs_branch_lengths(s=0.5, N=math.nan, n_diploids=4)

    for invalid_n in (2.9, True):
        with pytest.raises(ValueError, match="n_diploids must be an integer"):
            dgs_probabilities(s=0.5, n_diploids=invalid_n)

    with pytest.raises(ValueError, match="n_diploids"):
        dgs_probabilities(s=0.5, n_diploids=0)

    with pytest.raises(ValueError, match="method must be"):
        dgs_probabilities(s=0.5, n_diploids=2, method="unknown")


def test_recursive_n_diploids_above_practical_limit_warns(monkeypatch):
    def fake_cached(s, N, n_diploids, use_generation_time_scale, method):
        return (((n_diploids - 1, 1, 0), 1.0),)

    monkeypatch.setattr(
        probability_module,
        "_expected_dgs_branch_lengths_cached",
        fake_cached,
    )

    with pytest.warns(RuntimeWarning, match="may become slow"):
        branch_lengths = expected_dgs_branch_lengths(
            s=0.5,
            n_diploids=PRACTICAL_MAX_DIPLOIDS + 1,
            method="recursive",
        )

    assert branch_lengths[(PRACTICAL_MAX_DIPLOIDS, 1, 0)] == 1.0


def test_generation_time_scaling_remains_exposed():
    coalescent = expected_dgs_branch_lengths(
        s=0.5,
        N=500,
        n_diploids=2,
        use_generation_time_scale=False,
    )
    generations = expected_dgs_branch_lengths(
        s=0.5,
        N=500,
        n_diploids=2,
        use_generation_time_scale=True,
    )

    scale = (2.0 - 0.5) * 500
    for config, value in coalescent.items():
        assert generations[config] == pytest.approx(value * scale)


def test_n2_regression_probabilities_from_reference_values():
    probabilities = dgs_probabilities(s=0.5, n_diploids=2)

    assert probabilities == pytest.approx(
        {
            (0, 1, 1): 0.18604651162790695,
            (0, 2, 0): 0.09302325581395347,
            (1, 0, 1): 0.25581395348837205,
            (1, 1, 0): 0.46511627906976744,
        }
    )
