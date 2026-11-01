"""M-step of the MutSel prior (format see mutsel_rust/src/prior.rs) from the outputs (<prefix>.hmc.npz) of runs
under an old prior. Every alignment contributes its fitted log exchangeabilities, every site its posterior samples
with total weight 1:
- exchangeabilities: log S_ij ~ N(m_ij, sd^2), m = mean of the fitted log S over the alignments,
  sd^2 = mean squared deviation from m, exchangeability_reg = 1 / (2 sd^2). The fitted log S have mean 0 over
  the 190 entries (Mu is normalized and the prior fixes the scale), so m keeps that convention.
- site frequencies: weighted EM of the Gaussian mixture on the Helmert coordinates, started from the old mixture
- site_rate_sd: the M-step of SiteRatePrior::fit, sd^4 / 4 + sd^2 = mean of the squared log site rates

Usage: python3 fit_prior.py <old prior npz> <new prior npz> <hmc npz>...
"""

import argparse

import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import logsumexp

DIM = 19


def gmm_component_logpdf(x, weights, means, covs):
    out = np.empty((len(x), len(weights)))
    for k in range(len(weights)):
        chol = np.linalg.cholesky(covs[k])
        sol = solve_triangular(chol, (x - means[k]).T, lower=True)
        logdet = 2 * np.log(np.diag(chol)).sum()
        out[:, k] = np.log(weights[k]) - 0.5 * ((sol**2).sum(0) + logdet + DIM * np.log(2 * np.pi))
    return out


def fit_gmm(x, w, weights, means, covs, iterations, reg, tol):
    """Weighted EM of a full covariance Gaussian mixture on x [N, DIM] with weights w [N], from the given mixture"""
    prev = -np.inf
    for it in range(iterations):
        comp = gmm_component_logpdf(x, weights, means, covs)
        lse = logsumexp(comp, axis=1)
        ll = (w * lse).sum() / w.sum()
        print(f"GMM EM {it}: mean log density {ll:.6f}", flush=True)
        if ll - prev < tol:
            break
        prev = ll
        resp = np.exp(comp - lse[:, None]) * w[:, None]
        nk = resp.sum(0) + 1e-12
        weights = nk / nk.sum()
        means = resp.T @ x / nk[:, None]
        covs = np.stack([(resp[:, k, None] * (x - means[k])).T @ (x - means[k]) / nk[k] + reg * np.eye(DIM)
                         for k in range(len(nk))])
    return weights, means, covs


def site_rate_sd(log_rates):
    mean_square = np.mean(log_rates**2)
    return np.sqrt(2 * (np.sqrt(1 + mean_square) - 1))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("old_prior")
    parser.add_argument("new_prior")
    parser.add_argument("runs", nargs="+", help="hmc.npz outputs of runs under the old prior")
    parser.add_argument("--samples", type=int, default=10, help="posterior samples per site used for the mixture")
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--tol", type=float, default=1e-6, help="stop when the mean log density improves less")
    parser.add_argument("--reg", type=float, default=1e-4, help="added to the diagonal of the covariances")
    args = parser.parse_args()

    old = np.load(args.old_prior)
    log_S, coordinates, log_rates = [], [], []
    for path in args.runs:
        run = np.load(path)
        log_S.append(run["log_exchangeabilities"])
        samples = run["coordinate_samples"]  # [sites, samples, DIM]
        coordinates.append(samples[:, np.linspace(0, samples.shape[1] - 1, min(args.samples, samples.shape[1])).astype(int)])
        if "log_site_rates" in run:
            log_rates.append(run["log_site_rates"].ravel())
        print(f"{path}: {samples.shape[0]} sites", flush=True)

    log_S = np.array(log_S)  # [alignments, 190], entries (i, j), i > j, in row-major order
    mean_log_S = log_S.mean(0)
    exchangeability_sd = np.sqrt(np.mean((log_S - mean_log_S) ** 2))
    exchangeabilities = np.ones((20, 20))
    exchangeabilities[np.tril_indices(20, -1)] = np.exp(mean_log_S)  # tril_indices is row-major over i > j
    exchangeabilities = np.tril(exchangeabilities, -1) + np.tril(exchangeabilities, -1).T + np.eye(20)
    print(f"exchangeabilities: {len(log_S)} alignments, sd {exchangeability_sd:.4f} (old reg {float(old['exchangeability_reg'])})")

    rate_sd = site_rate_sd(np.concatenate(log_rates)) if log_rates else float(old["site_rate_sd"])
    print(f"site_rate_sd {rate_sd:.4f} (old {float(old['site_rate_sd'])})")

    coordinates = np.concatenate(coordinates)
    num_samples = coordinates.shape[1]
    x = coordinates.reshape(-1, DIM)
    w = np.full(len(x), 1 / num_samples)
    print(f"Gaussian mixture: {len(coordinates)} sites, {num_samples} samples per site, {len(old['weights'])} components")
    weights, means, covs = fit_gmm(x, w, old["weights"], old["means"], old["covs"], args.iterations, args.reg, args.tol)

    np.savez(
        args.new_prior,
        exchangeabilities=exchangeabilities,
        exchangeability_reg=np.array(1 / (2 * exchangeability_sd**2)),
        weights=weights,
        means=means,
        covs=covs,
        site_rate_sd=np.array(rate_sd),
    )
    print(f"saved {args.new_prior}")


if __name__ == "__main__":
    main()
