"""Readme example code"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from mpest import Distribution, MixtureDistribution, Problem
from mpest.em import EM
from mpest.em.breakpointers import StepCountBreakpointer
from mpest.em.distribution_checkers import FiniteChecker
from mpest.models import GaussianModel, WeibullModelExp
from mpest.optimizers import ScipyTNC

base_mixture_distribution = MixtureDistribution.from_distributions(
    [
        Distribution.from_params(WeibullModelExp, [0.5, 1.0]),
        Distribution.from_params(GaussianModel, [5.0, 1.0]),
    ],
    [0.33, 0.66],
)

x = base_mixture_distribution.generate(200)

problem = Problem(
    x,
    MixtureDistribution.from_distributions(
        [
            Distribution.from_params(WeibullModelExp, [1.0, 2.0]),
            Distribution.from_params(GaussianModel, [0.0, 5.0]),
        ]
    ),
)

em = EM(
    StepCountBreakpointer(max_step=8), FiniteChecker(), ScipyTNC(), method="likelihood"
)

result = em.solve(problem)


fig, axs = plt.subplots()
axs.set_xlabel("x")

sns.histplot(x, color="lightsteelblue")

axs_ = axs.twinx()
axs_.set_ylabel("p(x)")
axs_.set_yscale("log")

X = np.linspace(0.001, max(x), 2048)
axs_.plot(X, [base_mixture_distribution.pdf(x) for x in X], color="green", label="base")
axs_.plot(X, [result.content.pdf(x) for x in X], color="red", label="result")

plt.legend()
plt.show()
