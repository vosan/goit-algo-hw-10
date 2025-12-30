"""
Task 2. Compute a definite integral using the Monte Carlo method and
verify the result against an analytic value.

We keep the same function and bounds as in graph.py for consistency:
    f(x) = x^2 on [0, 2].

This script prints:
- Monte Carlo estimate with a standard error and a 95% confidence interval.
- Reference analytic value of the integral.
- Absolute and relative error of the Monte Carlo estimate.

Notes:
- The code uses NumPy's Generator for reproducibility.
"""
import math
from typing import Callable, Tuple

import numpy as np


def f(x: np.ndarray | float) -> np.ndarray | float:
    """Target function to integrate: f(x) = x^2.

    Works with scalars and NumPy arrays.
    """
    return np.asarray(x, dtype=float) ** 2


def integrate_monte_carlo(
    func: Callable[[np.ndarray], np.ndarray],
    a: float,
    b: float,
    n_samples: int = 200_000,
    rng_seed: int | None = 42,
) -> Tuple[float, float, Tuple[float, float]]:
    """Estimate the definite integral of func over [a, b] via Monte Carlo.

    The estimator is (b - a) * mean(func(U)), where U ~ Uniform(a, b).

    Args:
        func: Vectorized function accepting a NumPy array of x values.
        a: Lower integration bound.
        b: Upper integration bound.
        n_samples: Number of random samples to draw.
        rng_seed: Optional seed for reproducibility.

    Returns:
        A tuple (estimate, standard_error, ci_95), where ci_95 is a
        (low, high) 95% confidence interval using the normal approximation.
    """
    if n_samples <= 1:
        raise ValueError("n_samples must be greater than 1")

    rng = np.random.default_rng(rng_seed)
    xs = rng.uniform(a, b, size=n_samples)
    fx = func(xs)

    # Ensure NumPy array for consistent computations
    fx = np.asarray(fx, dtype=float)

    mean_fx = float(np.mean(fx))
    std_fx = float(np.std(fx, ddof=1))  # sample std for unbiased SE

    width = b - a
    estimate = width * mean_fx
    se = width * (std_fx / math.sqrt(n_samples))

    # 95% CI using normal quantile 1.96
    ci_low = estimate - 1.96 * se
    ci_high = estimate + 1.96 * se
    return estimate, se, (ci_low, ci_high)


def analytic_integral_x2(a: float, b: float) -> float:
    """Analytic integral of x^2 from a to b: (b^3 - a^3) / 3."""
    return (b**3 - a**3) / 3.0


def main() -> None:
    # Integration bounds matching graph.py
    a, b = 0.0, 2.0
    n = 200_000  # number of samples

    mc_estimate, mc_se, (ci_low, ci_high) = integrate_monte_carlo(f, a, b, n_samples=n, rng_seed=42)
    ref_value = analytic_integral_x2(a, b)

    abs_err = abs(mc_estimate - ref_value)
    rel_err = abs_err / ref_value if ref_value != 0 else float("nan")
    inside_ci = ci_low <= ref_value <= ci_high

    print("Monte Carlo method for the integral of f(x)=x^2 on [0, 2]")
    print(f"Number of samples: {n}")
    print(f"Integral estimate (MC): {mc_estimate:.8f}")
    print(f"Standard error (MC): {mc_se:.8f}")
    print(f"95% CI (MC): [{ci_low:.8f}, {ci_high:.8f}]")
    print()
    print(f"Reference value (analytic): {ref_value:.12f}")
    print(f"Absolute error (MC): {abs_err:.8e}")
    print(f"Relative error (MC): {rel_err:.6%}")
    verdict = "YES" if inside_ci else "NO"
    print(f"Does the reference value fall within the 95% MC CI? {verdict}")


if __name__ == "__main__":
    main()
