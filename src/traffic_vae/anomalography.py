"""Algorithm-2 updates for routed traffic with optional missing observations.

Reference: Mardani, Mateos, Giannakis, arXiv:1208.4043, equations (11)-(14).
Identity routing remains available for the original synthetic protocol.  The
paper's time-varying ``R_t`` and observation mask ``Omega_t`` are accepted per
sample; the default remains a fixed routing matrix with all links observed.
A normal-only SVD prefix initializes full-rank sufficient statistics; this is an
explicit adaptation, not the paper's random P / zero-statistics initialization.
"""
from dataclasses import dataclass
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso


def solve_lasso(design, target, penalty, max_iterations=3000, tolerance=1e-7):
    """Solve 0.5 ||target - design @ a||^2 + penalty ||a||_1 and check KKT."""
    design = np.asarray(design, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if (design.ndim != 2 or min(design.shape) < 1 or target.shape != (len(design),)
            or not np.isfinite(design).all() or not np.isfinite(target).all()
            or not np.isfinite([penalty, tolerance]).all() or min(penalty, tolerance) <= 0
            or type(max_iterations) is not int or max_iterations < 1):
        raise ValueError('Lasso requires a finite design, target, and positive solver settings.')
    gradient_at_zero = design.T @ target
    iterations, warning_seen = 0, False
    if np.max(np.abs(gradient_at_zero)) <= penalty:
        anomaly = np.zeros(design.shape[1])
    else:
        # sklearn averages squared loss; divide alpha by the number of design rows.
        lasso = Lasso(alpha=penalty / len(target), fit_intercept=False,
                      max_iter=max_iterations, tol=tolerance, selection='cyclic')
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always', ConvergenceWarning)
            lasso.fit(design, target)
        warning_seen = any(issubclass(item.category, ConvergenceWarning) for item in caught)
        anomaly = lasso.coef_
        iterations = int(lasso.n_iter_)
    gradient = design.T @ (design @ anomaly - target)
    violation = np.where(anomaly != 0, np.abs(gradient + penalty * np.sign(anomaly)),
                         np.maximum(np.abs(gradient) - penalty, 0))
    relative_kkt = float(np.max(violation) / max(1., np.max(np.abs(gradient_at_zero))))
    if not np.isfinite(anomaly).all() or not np.isfinite(relative_kkt):
        raise FloatingPointError('Non-finite Lasso solution.')
    return anomaly, {'lasso_iterations': iterations, 'relative_kkt_residual': relative_kkt,
                     'converged': not warning_seen and relative_kkt <= max(1e-5, 100 * tolerance)}


@dataclass
class OnlineResult:
    low_rank: np.ndarray
    sparse: np.ndarray
    history: list[dict]


class OnlineAnomalography:
    def __init__(self, rank: int, ridge: float, sparse_penalty: float,
                 max_iterations: int = 3000, tolerance: float = 1e-7, *,
                 routing: np.ndarray | None = None, forgetting_factor: float = 1.,
                 output_after_update: bool = False):
        if (type(rank) is not int or rank < 1 or type(max_iterations) is not int
                or max_iterations < 1 or not np.isfinite([ridge, sparse_penalty, tolerance]).all()
                or min(ridge, sparse_penalty, tolerance) <= 0):
            raise ValueError('Invalid online anomalography parameters.')
        self.rank = rank
        self.ridge = ridge
        self.sparse_penalty = sparse_penalty
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        if not np.isfinite(forgetting_factor) or not 0 < forgetting_factor <= 1:
            raise ValueError('forgetting_factor must lie in (0, 1].')
        self.forgetting_factor = forgetting_factor
        self.output_after_update = output_after_update
        self.routing = None if routing is None else np.asarray(routing, dtype=np.float64).copy()
        if self.routing is not None and (self.routing.ndim != 2 or min(self.routing.shape) < 1
                                         or not np.isfinite(self.routing).all()):
            raise ValueError('routing must be a finite links-by-flows matrix.')
        self.basis = None
        self.gram = None
        self.cross = None

    def initialize(self, normal_prefix: np.ndarray):
        values = np.asarray(normal_prefix, dtype=np.float64)
        if (values.ndim != 2 or min(values.shape) < self.rank
                or not np.isfinite(values).all()):
            raise ValueError('Warm-up matrix must be finite with both dimensions >= rank.')
        if self.routing is None:
            self.routing = np.eye(values.shape[1])
        if self.routing.shape[0] != values.shape[1]:
            raise ValueError('Routing rows must match the number of observed links.')
        u, singular, _ = np.linalg.svd(values, full_matrices=False)
        if singular[self.rank - 1] <= 1e-10:
            raise ValueError('Warm-up matrix does not span the requested rank.')
        coordinates = u[:, :self.rank] * np.sqrt(singular[:self.rank])
        shared_gram = coordinates.T @ coordinates
        self.gram = np.repeat(shared_gram[None, :, :], values.shape[1], axis=0)
        self.cross = values.T @ coordinates
        self.basis = np.asarray([
            np.linalg.solve(item + self.ridge * np.eye(self.rank), target)
            for item, target in zip(self.gram, self.cross)
        ])

    def copy(self):
        clone = OnlineAnomalography(self.rank, self.ridge, self.sparse_penalty,
                                    self.max_iterations, self.tolerance, routing=self.routing,
                                    forgetting_factor=self.forgetting_factor,
                                    output_after_update=self.output_after_update)
        clone.basis = self.basis.copy()
        clone.gram = self.gram.copy()
        clone.cross = self.cross.copy()
        return clone

    def coefficients(self, observation, *, routing=None, mask=None):
        """Solve the joint ridge-coordinate/L1-anomaly subproblem at fixed P."""
        y = np.asarray(observation, dtype=np.float64)
        if self.basis is None or y.shape != (len(self.basis),) or not np.isfinite(y).all():
            raise ValueError('Initialize the tracker and supply a finite feature vector.')
        current_routing = self.routing if routing is None else np.asarray(routing, dtype=np.float64)
        if (current_routing.ndim != 2 or current_routing.shape[0] != len(y)
                or current_routing.shape[1] != self.routing.shape[1]
                or not np.isfinite(current_routing).all()):
            raise ValueError('routing must have the initialized links-by-flows shape.')
        observed = np.ones(len(y), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
        if observed.shape != (len(y),) or not observed.any():
            raise ValueError('mask must select at least one observed link.')
        observed_basis = self.basis[observed]
        observed_routing = current_routing[observed]
        observed_y = y[observed]
        projection = np.linalg.solve(
            self.ridge * np.eye(self.rank) + observed_basis.T @ observed_basis,
            observed_basis.T,
        )
        # F.T @ F = I - P @ (lambda I + P.T @ P)^(-1) @ P.T.
        transform = np.vstack([np.eye(len(observed_y)) - observed_basis @ projection,
                               np.sqrt(self.ridge) * projection])
        anomaly, diagnostics = solve_lasso(transform @ observed_routing, transform @ observed_y,
                                           self.sparse_penalty, self.max_iterations, self.tolerance)
        coordinates = projection @ (observed_y - observed_routing @ anomaly)
        nominal = self.basis @ coordinates
        return nominal, anomaly, coordinates, diagnostics

    def process(self, values: np.ndarray, update: bool = True, *,
                routings: np.ndarray | None = None,
                masks: np.ndarray | None = None) -> OnlineResult:
        """Estimate flow anomalies and update causally; nominal-output timing is explicit."""
        values = np.asarray(values, dtype=np.float64)
        if (self.basis is None or values.ndim != 2 or values.shape[1] != len(self.basis)
                or not np.isfinite(values).all()):
            raise ValueError('Initialize the tracker and supply finite time-by-link observations.')
        if routings is None:
            routing_values = None
        else:
            routing_values = np.asarray(routings, dtype=np.float64)
            if (routing_values.shape != (len(values), *self.routing.shape)
                    or not np.isfinite(routing_values).all()):
                raise ValueError('routings must be time-by-links-by-flows.')
        if masks is None:
            mask_values = np.ones_like(values, dtype=bool)
        else:
            mask_values = np.asarray(masks, dtype=bool)
            if mask_values.shape != values.shape or not mask_values.any(axis=1).all():
                raise ValueError('masks must be time-by-link and retain an observation per time.')
        low_rank = np.empty_like(values)
        sparse = np.empty((len(values), self.routing.shape[1]))
        history = []
        for index, y in enumerate(values):
            routing = self.routing if routing_values is None else routing_values[index]
            mask = mask_values[index]
            nominal, anomaly, q, diagnostics = self.coefficients(y, routing=routing, mask=mask)
            low_rank[index], sparse[index] = nominal, anomaly
            history.append({'time_index': index, 'observed_links': int(mask.sum()), **diagnostics})
            if update:
                self.gram *= self.forgetting_factor
                self.cross *= self.forgetting_factor
                residual = y - routing @ anomaly
                outer = np.outer(q, q)
                self.gram[mask] += outer
                self.cross[mask] += residual[mask, None] * q
                for link in np.flatnonzero(mask):
                    self.basis[link] = np.linalg.solve(
                        self.gram[link] + self.ridge * np.eye(self.rank), self.cross[link]
                    )
                if self.output_after_update:
                    low_rank[index] = self.basis @ q
        return OnlineResult(low_rank, sparse, history)
