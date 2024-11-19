""" A module that provides various metrics for evaluating the quality of parameter estimation. """

from abc import ABC, abstractmethod
from itertools import permutations

import numpy as np

from mpest import MixtureDistribution
from mpest.utils import ANamed


class AMetric(ANamed, ABC):
    """
    Abstract class of metric
    """

    @abstractmethod
    def _single_error(
        self, base_mixture: MixtureDistribution, result_mixture: MixtureDistribution
    ) -> float:
        """
        A function for calculating the error for one pair of mixtures
        """

    @abstractmethod
    def error(
        self,
        base_mixtures: list[MixtureDistribution],
        result_mixtures: list[MixtureDistribution],
    ) -> list[float]:
        """
        A public function for calculating the error of multiple pairs of mixtures
        """


class MSE(AMetric):
    """
    The class that calculates the MSE
    """

    @property
    def name(self) -> str:
        return "MSE"

    def _single_error(self, base_mixture, result_mixture):
        pass

    def error(self, base_mixtures, result_mixtures):
        return sum(
            self._single_error(base, result)
            for base, result in zip(base_mixtures, result_mixtures)
        ) / len(result_mixtures)


class Parametric(AMetric):
    """
    A class that calculates the absolute difference in the parameters of mixtures. Does not use the difference of a prior probabilities
    """

    @property
    def name(self) -> str:
        return "ParamsError"

    def _single_error(self, base_mixture, result_mixture):
        base_p, res_p = (
            [d.params for d in ld] for ld in (base_mixture, result_mixture)
        )

        param_diff = min(
            sum(sum(abs(x - y)) for x, y in zip(base_p, _res_p))
            for _res_p in permutations(res_p)
        )

        return param_diff

    def error(self, base_mixtures, result_mixtures):
        return sum(
            self._single_error(base, result)
            for base, result in zip(base_mixtures, result_mixtures)
        ) / len(result_mixtures)


class PDFMetric(AMetric):
    """
    A class that calculates the absolute difference in the densities of mixtures at different points
    """

    def __init__(self, a, b):
        self.start = a
        self.end = b

    @property
    def name(self) -> str:
        return "PDFError"

    def _single_error(
        self, base_mixture: MixtureDistribution, result_mixture: MixtureDistribution
    ) -> float:
        x_linspace = np.linspace(self.start, self.end, 30)
        errors = [abs(base_mixture.pdf(x) - result_mixture.pdf(x)) for x in x_linspace]

        return sum(errors)

    def error(self, base_mixtures, result_mixtures):
        return sum(
            self._single_error(base, result)
            for base, result in zip(base_mixtures, result_mixtures)
        ) / len(result_mixtures)


METRICS = {
    MSE().name: MSE,
    Parametric().name: Parametric,
    PDFMetric(-np.inf, np.inf).name: PDFMetric,
}
