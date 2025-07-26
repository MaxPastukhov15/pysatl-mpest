"""Module for research on uniform distribution mixtures with k components."""

from typing import Union

import numpy as np

from mpest.core.mixture_distribution import DistributionInMixture, MixtureDistribution
from mpest.core.problem import Problem
from mpest.em.methods.abstract_steps import AExpectation, AMaximization
from mpest.utils import ResultWithError


class UniformEStep(AExpectation):
    """
    E-step for uniform mixtures using Bayesian responsibilities.
    Calculates posterior probabilities for each sample belonging to each component.
    """

    def step(self, problem: Problem) -> Union[tuple[Problem, np.ndarray], ResultWithError]:
        mixture = problem.distributions
        samples = problem.samples

        # Calculate unnormalized matrix of hidden variables
        h = np.array(
            [dist.prior_probability * np.array([dist.model.pdf(x, dist.params) for x in samples]) for dist in mixture]
        )

        # Normalize responsibilities
        sum_h = np.sum(h, axis=0)
        sum_h[sum_h == 0] = 1.0  # Avoid division by zero
        h = h / sum_h

        return problem, h.T


class UniformMStep(AMaximization):
    """
    M-step for uniform distributions using maximum likelihood estimation.
    Updates component parameters based on assigned samples.
    """

    def step(self, e_result: Union[tuple[Problem, np.ndarray], ResultWithError]) -> ResultWithError:
        if isinstance(e_result, ResultWithError):
            return e_result

        problem, responsibilities = e_result
        samples = problem.samples
        mixture = problem.distributions

        new_distributions = []
        for i, dist in enumerate(mixture):
            resp = responsibilities[:, i]
            mask = resp > 0
            weighted_samples = samples[mask]

            if len(weighted_samples) == 0:
                # No samples assigned to this component - keep old params
                new_distributions.append(dist)
                continue

            # ML estimates for uniform are min and max of assigned samples
            a = np.min(weighted_samples)
            b = np.max(weighted_samples)

            
            if a >= b:
                a = (dist.params[0] + dist.params[1]) / 2 - 0.05 
                b = (dist.params[0] + dist.params[1]) / 2 + 0.05

            prior = np.mean(resp) 
            new_dist = DistributionInMixture(dist.model, np.array([a, b], dtype=np.float64), prior)
            new_distributions.append(new_dist)

        return ResultWithError(MixtureDistribution(new_distributions))


class UniformLMomentsMStep(AMaximization):
    """
    M-step for uniform mixtures using L-moments estimation.
    Provides more robust parameter estimates than ML in some cases(gap).
    """

    def _calculate_lmoments(self, samples: np.ndarray, weights: np.ndarray) -> tuple[float, float] | None:
        """Calculate first two L-moments from weighted samples"""
        if len(samples) < 2:
            return None

        # Sort samples and corresponding weights
        sorted_idx = np.argsort(samples)
        sorted_samples = samples[sorted_idx]
        sorted_weights = weights[sorted_idx]

        # Normalize weights
        sorted_weights = sorted_weights / np.sum(sorted_weights)

        # Calculate probability weighted moments
        cum_weights = np.cumsum(sorted_weights)
        b0 = np.sum(sorted_weights * sorted_samples)
        b1 = np.sum(sorted_weights * (1 - cum_weights + sorted_weights / 2) * sorted_samples)

        # Convert to L-moments
        l1 = b0
        l2 = 2 * b1 - b0

        return l1, l2

    def step(self, e_result: Union[tuple[Problem, np.ndarray], ResultWithError]) -> ResultWithError:
        if isinstance(e_result, ResultWithError):
            return e_result

        problem, responsibilities = e_result
        samples = problem.samples
        mixture = problem.distributions

        new_distributions = []
        for i, dist in enumerate(mixture):
            resp = responsibilities[:, i]
            mask = resp > 0
            weighted_samples = samples[mask]
            weights = resp[mask]

            if len(weighted_samples) == 0:
                # No samples assigned to this component - keep old params
                new_distributions.append(dist)
                continue

            # Calculate L-moments
            moments = self._calculate_lmoments(weighted_samples, weights)
            if moments is None:
                new_distributions.append(dist)
                continue

            # Use the LMomentsParameterMixin from Uniform model
            a = dist.model.calc_alpha(moments)  # type: ignore
            b = dist.model.calc_beta(moments)  # type: ignore

            # Ensure valid parameters (a < b)
            a, b = min(a, b), max(a, b)

            # Update prior probability
            prior = np.mean(resp)

            new_dist = DistributionInMixture(dist.model, np.array([a, b], dtype=np.float64), prior)
            new_distributions.append(new_dist)

        return ResultWithError(MixtureDistribution(new_distributions))
