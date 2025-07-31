"""Module for research on uniform distribution mixtures with k components."""

from typing import Union

import numpy as np

from mpest.core.mixture_distribution import DistributionInMixture, MixtureDistribution
from mpest.core.problem import Problem
from mpest.em.methods.abstract_steps import AMaximization
from mpest.utils import ResultWithError


class UniformMStep(AMaximization):
    """
    M-step for uniform distributions using maximum likelihood estimation.
    Updates component parameters based on assigned samples.
    """

    EPSILON: float = 1e-6  # to avoid zero width uniform distributions

    def step(self, e_result: Union[tuple[Problem, np.ndarray], ResultWithError]) -> ResultWithError:
        if isinstance(e_result, ResultWithError):
            return e_result

        problem, responsibilities = e_result
        samples = problem.samples
        mixture = problem.distributions

        new_weights = np.mean(responsibilities, axis=0)

        new_distributions = []
        for i, dist in enumerate(mixture):
            weights = responsibilities[:, i]

            if np.sum(weights) < self.EPSILON:
                new_a = np.percentile(samples, 10) + np.random.uniform(-0.1, 0.1)
                new_b = new_a + np.abs(np.percentile(samples, 90) - new_a) + self.EPSILON
            else:
                new_a = np.percentile(samples, 5, weights=weights)
                new_b = np.percentile(samples, 95, weights=weights)
                new_b = np.max(new_a + self.EPSILON, new_b)

            new_a = np.clip(new_a, np.min(samples), np.max(samples) - self.EPSILON)
            new_b = np.clip(new_b, new_a + self.EPSILON, np.max(samples))

            new_dist = DistributionInMixture(dist.model, np.array([new_a, new_b], dtype=np.float64), new_weights[i])
            new_distributions.append(new_dist)

        return ResultWithError(MixtureDistribution(new_distributions))


class UniformLMomentsMStep(AMaximization):
    """
    M-step for uniform mixtures using L-moments estimation.
    Provides more robust parameter estimates than ML in some cases(gap).
    """

    def _calculate_lmoments(self, samples: np.ndarray, weights: np.ndarray) -> tuple[float, float] | None:
        """Calculate first two L-moments from weighted samples"""
        if len(samples) == 0:
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
            a = dist.model.calc_alpha(moments)
            b = dist.model.calc_beta(moments)

            # Ensure valid parameters (a < b)
            a, b = min(a, b), max(a, b)
            if not np.isfinite(a) or not np.isfinite(b):
                a, b = np.min(samples), np.max(samples)

            # Update prior probability
            prior = np.mean(resp)

            new_dist = DistributionInMixture(dist.model, np.array([a, b], dtype=np.float64), prior)
            new_distributions.append(new_dist)

        return ResultWithError(MixtureDistribution(new_distributions))
