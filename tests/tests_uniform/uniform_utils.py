"""Module to run tests"""

from mpest.core.problem import Problem, Result
from mpest.em import EM
from mpest.em.breakpointers import ParamDifferBreakpointer, StepCountBreakpointer
from mpest.em.distribution_checkers import (
    FiniteChecker,
    PriorProbabilityThresholdChecker,
)
from mpest.em.methods.method import Method
from mpest.em.methods.uniform_estimator import UniformEStep, UniformLMomentsMStep, UniformMStep


def run_test_ML(problem: Problem, deviation: float) -> Result:
    """TODO"""
    method = Method(UniformEStep(), UniformMStep())
    em_algo = EM(
        StepCountBreakpointer() + ParamDifferBreakpointer(deviation=deviation),
        FiniteChecker() + PriorProbabilityThresholdChecker(),
        method,
    )

    return em_algo.solve(problem=problem)


def run_test_LMoments(problem: Problem, deviation: float) -> Result:
    """TODO"""
    method = Method(UniformEStep(), UniformLMomentsMStep())
    em_algo = EM(
        StepCountBreakpointer() + ParamDifferBreakpointer(deviation=deviation),
        FiniteChecker() + PriorProbabilityThresholdChecker(),
        method,
    )

    return em_algo.solve(problem=problem)
