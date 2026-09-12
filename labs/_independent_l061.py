"""Independent L061 checks against sklearn GP prediction and density quadrature.

These checks exercise the lesson implementation using separate numerical methods.
They establish local operator behavior, not reproduction of the paper's figures.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from scipy.integrate import quad
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel


def check():
    from relkit import pfn_l061_v2 as m
    torch.set_num_threads(1)
    rng = np.random.default_rng(61013)
    records = []
    for features in (1, 2, 5):
        for count in (1, 3, 11):
            for duplicate in (False, True):
                x = rng.uniform(size=(count + 7, features))
                if duplicate and count > 1:
                    x[1] = x[0]
                y = rng.normal(size=count)
                reference = GaussianProcessRegressor(
                    kernel=RBF(.6, length_scale_bounds='fixed')
                    + WhiteKernel(1e-4, noise_level_bounds='fixed'),
                    alpha=0., optimizer=None, normalize_y=False).fit(x[:count], y)
                mu, sd = reference.predict(x[count:], return_std=True)
                actual_mu, actual_var = m.gp_posterior(
                    torch.tensor(x[None], dtype=torch.float64),
                    torch.tensor(y[None], dtype=torch.float64), count)
                mean_error = float(np.max(np.abs(actual_mu.numpy()[0] - mu)))
                variance_error = float(np.max(np.abs(actual_var.numpy()[0] - sd ** 2)))
                assert mean_error < 1e-8 and variance_error < 1e-9
                records.append(dict(features=features, context=count, duplicate=duplicate,
                                    mean_error=mean_error, variance_error=variance_error))
    borders = torch.tensor([-3., -1., 0., 2., 5.], dtype=torch.float64)
    probabilities = np.array([.1, .2, .3, .4])
    logits = torch.tensor(np.log(probabilities)[None], dtype=torch.float64)
    def density(y):
        return float(torch.exp(-m.riemann_nll(logits, torch.tensor([y], dtype=torch.float64), borders)))
    intervals = [(-np.inf, -1.), (-1., 0.), (0., 2.), (2., np.inf)]
    masses = [quad(density, a, b, epsabs=1e-11, epsrel=1e-11)[0] for a, b in intervals]
    np.testing.assert_allclose(masses, probabilities, atol=1e-10, rtol=0.)
    expected_mean = sum(quad(lambda y: y * density(y), a, b, epsabs=1e-10)[0]
                        for a, b in intervals)
    actual_mean = float(m.riemann_mean(logits, borders))
    assert abs(actual_mean - expected_mean) < 1e-10
    cdf_errors = []
    for point in (-12., -3., -1., -.5, 0., 1., 2., 5., 12.):
        integral = sum(quad(density, a, min(point, b), epsabs=1e-11)[0]
                       for a, b in intervals if point > a)
        actual = float(m.riemann_cdf(logits, torch.tensor([point], dtype=torch.float64), borders))
        cdf_errors.append(abs(integral - actual))
    assert max(cdf_errors) < 1e-10
    report = dict(status='PASS', gp_fixtures=records, integrated_bin_masses=masses,
                  integrated_predictive_mean=expected_mean,
                  predictive_mean_error=abs(actual_mean - expected_mean),
                  max_cdf_integral_error=max(cdf_errors),
                  source_sha256=hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest(),
                  scope='Independent sklearn noisy GP posterior, duplicate-feature conditioning, '
                        'and numerical integration of full-support density; no paper-training claim')
    path = Path(__file__).resolve().parents[1] / 'reviews/lesson-quality-audit-047-070/061-independent.json'
    path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    check()
