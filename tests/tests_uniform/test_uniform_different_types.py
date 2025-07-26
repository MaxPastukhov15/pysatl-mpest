"""Module for unit tests on uniform_estimator"""

# pylint: disable=duplicate-code
# pylint: disable=too-many-arguments
# pylint: disable=too-many-locals

import numpy as np
import pytest
from mpest.core.distribution import Distribution
from mpest.core.mixture_distribution import MixtureDistribution
from mpest.core.problem import Problem
from mpest.models import AModelWithGenerator, Uniform
from mpest.utils import Factory

from tests.tests_uniform.uniform_utils import run_test_LMoments, run_test_ML
from tests.utils import (
    check_for_params_error_tolerance,
    check_for_priors_error_tolerance,
)


def idfunc(vals):
    """Function for customizing pytest ids"""
    if isinstance(vals, Factory):
        return vals.cls().name
    if isinstance(vals, list):
        return vals
    return f"{vals}"


@pytest.mark.parametrize(
    "model_factory, params, start_params, prior_probability, size, deviation,"
    "expected_params_error, expected_priors_error",
    [
        # 1. Without overlap and gap
        (
            Factory(Uniform),
            [(0.0, 1.0), (2.0, 3.0)],
            [(0.5, 1.5), (1.5, 2.5)],
            [0.5, 0.5],
            500,
            0.01,
            0.3,
            0.3,
        ),
        # 2. With overlap
        (
            Factory(Uniform),
            [(0.5, 1.5), (1.0, 2.0)],
            [(0.0, 1.0), (1.5, 2.5)],
            [0.6, 0.4],
            500,
            0.01,
            0.3,
            0.3,
        ),
        # 3. With gap
        (
            Factory(Uniform),
            [(0.0, 1.0), (3.0, 4.0)],
            [(-0.5, 1.5), (2.5, 3.5)],
            [0.7, 0.3],
            500,
            0.01,
            0.3,
            0.3,
        ),
    ],
    ids=["no_overlap_no_gap", "with_overlap", "with_gap"],
)
def test_uniform_mixture_types(
    model_factory: Factory[AModelWithGenerator],
    params,
    start_params,
    prior_probability: list[float],
    size: int,
    deviation: float,
    expected_params_error,
    expected_priors_error,
):
    """Testing EM-algorithms for mixture uniform distribution:
    - without overlap and gap,
    - with overlap,
    - with gap.
    """
    np.random.seed(42)

    models = [model_factory.construct() for _ in range(len(params))]
    params = [np.array(param) for param in params]
    start_params = [np.array(param) for param in start_params]

    # Creating actual mixture
    true_mixture = MixtureDistribution.from_distributions(
        [Distribution(model, param) for model, param in zip(models, params)],
        prior_probability,
    )

    # Generating sample
    samples = true_mixture.generate(size)

    # Creating problem
    problem = Problem(
        samples=samples,
        distributions=MixtureDistribution.from_distributions(
            [Distribution(model, param) for model, param in zip(models, start_params)]
        ),
    )

    # Testing both methods: ML and L-moments run_test_LMoments
    result = run_test_ML(problem=problem, deviation=deviation)
    assert check_for_params_error_tolerance([result], true_mixture, expected_params_error)
    assert check_for_priors_error_tolerance([result], true_mixture, expected_priors_error)
