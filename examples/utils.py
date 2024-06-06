"""Module which provides many useful utils for improving code writing experience"""

from typing import NamedTuple, Callable
from functools import partial

import random
import time
import pickle
import numpy as np
from tqdm.contrib.concurrent import process_map

from mpest.types import Samples
from mpest.mixture_distribution import MixtureDistribution
from mpest.problem import Problem, Result
from mpest.em import EM
from mpest.em.breakpointers import StepCountBreakpointer, ParamDifferBreakpointer
from mpest.em.distribution_checkers import (
    FiniteChecker,
    PriorProbabilityThresholdChecker,
)
from mpest.optimizers import TOptimizer

from examples.config import RESULTS_FOLDER

np.seterr(all="ignore")


class Test(NamedTuple):
    """NamedTuple which represents all needed test data"""

    index: int
    all_data: Samples
    true_mixture: MixtureDistribution

    problem: Problem
    solvers: list[EM]

    runs: int


class SingleSolverResult(NamedTuple):
    """NamedTuple which represents all needed single EM solver data"""

    test: Test

    solver: EM
    result: Result
    steps: int
    time: float

    log: list[EM.Log.Item] = []


class TestResult(NamedTuple):
    """NamedTuple which represents test result"""

    test: Test
    results: list[SingleSolverResult]


class Clicker:
    """Class which allows you to \"click\" """

    # pylint: disable=too-few-public-methods

    def __init__(self) -> None:
        self._counter = -1

    def click(self):
        """Click method"""
        self._counter += 1
        return self._counter


def run_test(
    test: Test,
    create_history=False,
    remember_time=False,
) -> TestResult:
    """Runs given test and optional creates logs"""

    times = []
    results = []

    for solver in test.solvers:
        for _ in range(test.runs):
            start = time.perf_counter()
            result = solver.solve_logged(
                test.problem,
                create_history=create_history,
                remember_time=remember_time,
            )
            stop = time.perf_counter()
            times.append(stop - start)

        results.append(
            SingleSolverResult(
                test,
                solver,
                result.result,
                result.log.steps,
                float(np.mean(times)),
                result.log.log,
            )
        )

    return TestResult(test, results)


def run_tests(
    tests: list[Test],
    workers_count: int,
    shuffled: bool = True,
    chunksize: int = 1,
    create_history=False,
    remember_time=False,
) -> list[TestResult]:
    """Runs given tests multithreaded and optional creates logs"""

    # pylint: disable=too-many-arguments

    if not shuffled:
        _tests = tests
    else:
        _tests = list(tests)
        random.shuffle(_tests)

    results: list[TestResult] = process_map(
        partial(run_test, create_history=create_history, remember_time=remember_time),
        _tests,
        max_workers=workers_count,
        chunksize=chunksize,
    )

    if shuffled:
        key: Callable[[TestResult], int] = lambda t: t.test.index
        results.sort(key=key)

    return results


def save_results(results: list[TestResult], name: str) -> None:
    """Saves test results into standard folder using pickle"""
    with open(RESULTS_FOLDER / f"{name}.pkl", "wb") as f:
        pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)


def open_results(name: str) -> list[TestResult]:
    """Loads test results from standard folder using pickle"""

    with open(RESULTS_FOLDER / f"{name}.pkl", "rb") as f:
        return pickle.load(f)


def init_solver(
    max_step: int | None,
    max_deviation: float | None,
    prior_probability_threshold: float | None,
    prior_probability_threshold_step: int | None,
    optimizer: TOptimizer,
):
    """TODO"""

    breakpointer = StepCountBreakpointer(max_step)
    if max_deviation is not None:
        breakpointer += ParamDifferBreakpointer(max_deviation)
    return EM(
        breakpointer,
        FiniteChecker()
        + PriorProbabilityThresholdChecker(
            prior_probability_threshold, prior_probability_threshold_step
        ),
        optimizer,
    )
