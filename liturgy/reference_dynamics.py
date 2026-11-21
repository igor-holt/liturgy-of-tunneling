"""Exactly sampled overdamped Ornstein-Uhlenbeck reference model.

This is a classical, equilibrium Langevin toy model. It is not an M-theory,
quantum decoherence, or biological/clinical simulator. In particular, its
fluctuation-dissipation relationship is a stated assumption, not a consequence
of geometric compactification.

  dX = -(stiffness/friction) X dt + sqrt(2*kBT/friction) dW.

The invariant equilibrium variance is kBT/stiffness.
"""

from dataclasses import dataclass
from math import exp, expm1, isfinite, sqrt
from random import Random


@dataclass(frozen=True)
class OUParameters:
    stiffness: float
    friction: float
    thermal_energy: float
    dt: float

    def __post_init__(self) -> None:
        vals = (self.stiffness, self.friction, self.thermal_energy, self.dt)
        if not all(isfinite(v) for v in vals):
            raise ValueError("All parameters must be finite")
        if self.stiffness <= 0 or self.friction <= 0 or self.dt <= 0:
            raise ValueError("Stiffness, friction and dt must be strictly positive")
        if self.thermal_energy < 0:
            raise ValueError("Thermal energy must be nonnegative")

    @property
    def decay(self) -> float:
        return exp(-self.stiffness * self.dt / self.friction)

    @property
    def equilibrium_variance(self) -> float:
        return self.thermal_energy / self.stiffness

    @property
    def innovation_variance(self) -> float:
        ratio = self.stiffness * self.dt / self.friction
        return self.equilibrium_variance * (-expm1(-2.0 * ratio))


def step(x: float, params: OUParameters, rng: Random) -> float:
    """Advance the OU process exactly for a fixed time step."""
    if not isfinite(x):
        raise ValueError("State must be finite")
    return params.decay * x + sqrt(params.innovation_variance) * rng.gauss(0.0, 1.0)


def second_moment(x0: float, n_steps: int, params: OUParameters) -> float:
    """Analytic E[X_n**2] for fixed deterministic initial state."""
    if not isfinite(x0):
        raise ValueError("Initial state must be finite")
    if isinstance(n_steps, bool) or not isinstance(n_steps, int) or n_steps < 0:
        raise ValueError("n_steps must be a nonnegative integer")
    a2n = exp(-2.0 * params.stiffness * params.dt * n_steps / params.friction)
    return x0 * x0 * a2n + params.equilibrium_variance * (1.0 - a2n)
