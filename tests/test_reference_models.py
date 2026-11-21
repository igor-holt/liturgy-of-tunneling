"""Stdlib-only independent software checks for safe reference models."""

import random
import unittest

from liturgy.reference_dynamics import OUParameters, second_moment, step
from liturgy.schedule import skip_indices


class TestOrnsteinUhlenbeck(unittest.TestCase):
    def setUp(self):
        self.p = OUParameters(stiffness=2.0, friction=4.0, thermal_energy=3.0, dt=0.5)

    def test_fdt_invariant_variance(self):
        self.assertAlmostEqual(self.p.equilibrium_variance, 1.5)
        self.assertAlmostEqual(
            self.p.innovation_variance,
            self.p.equilibrium_variance * (1.0 - self.p.decay**2),
        )

    def test_initial_and_stationary_second_moment(self):
        self.assertEqual(second_moment(2.0, 0, self.p), 4.0)
        self.assertAlmostEqual(second_moment(2.0, 1000, self.p), 1.5, places=10)

    def test_zero_temperature_is_deterministic(self):
        p = OUParameters(1.0, 1.0, 0.0, 0.1)
        self.assertAlmostEqual(step(2.0, p, random.Random(1)), 2.0 * p.decay)

    def test_fixed_seed_reproducible(self):
        def trajectory():
            rng = random.Random(314159)
            x = 1.0
            for _ in range(10):
                x = step(x, self.p, rng)
            return x
        self.assertEqual(trajectory(), trajectory())

    def test_monte_carlo_second_moment(self):
        rng = random.Random(10)
        target = second_moment(0.0, 30, self.p)
        trials = 7000
        accumulator = 0.0
        for _ in range(trials):
            x = 0.0
            for _ in range(30):
                x = step(x, self.p, rng)
            accumulator += x * x
        self.assertLess(abs(accumulator / trials - target) / target, 0.06)

    def test_reject_bad_parameters_and_states(self):
        for bad in (0, -1, float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                OUParameters(bad, 1, 1, 1)
        with self.assertRaises(ValueError):
            second_moment(0.0, -1, self.p)
        with self.assertRaises(ValueError):
            step(float("nan"), self.p, random.Random(1))


class TestHostSchedule(unittest.TestCase):
    def test_exact_three_events(self):
        self.assertEqual(skip_indices(depth=137, prime=17, event_count=3), (17, 34, 51))

    def test_zero_events(self):
        self.assertEqual(skip_indices(depth=1, prime=17, event_count=0), ())

    def test_boundaries(self):
        self.assertEqual(skip_indices(depth=137, prime=17, event_count=8)[-1], 136)
        with self.assertRaises(ValueError):
            skip_indices(depth=137, prime=17, event_count=9)

    def test_reject_nonprime_or_invalid_types(self):
        for triple in ((137, 16, 3), (137, 1, 1), (True, 17, 0), (137, 17, -1)):
            with self.assertRaises(ValueError):
                skip_indices(*triple)


if __name__ == "__main__":
    unittest.main()
