import argparse
import pandas as pd
from functools import partial

import numpy as np
from jax import lax, random
import jax.numpy as jnp

import optax
from optax import piecewise_constant_schedule

import numpyro
from numpyro.util import enable_x64
import numpyro.distributions as dist
from numpyro.infer import Predictive, RenyiELBO
from numpyro.infer.autoguide import (
    AutoBNAFNormal,
    AutoDAIS,
    AutoDiagonalNormal,
    AutoMultivariateNormal,
)
from numpyro.infer.initialization import init_to_median
from numpyro.util import fori_loop

from rvrs import SVI, Trace_ELBO, AutoRVRS


init_loc_fn = init_to_median(num_samples=10)


def svi_body_fn(svi, i, val):
    svi_state, loss = svi.update(val)
    return svi_state


def logistic_model(X, Y):
    coef = numpyro.sample("coef", dist.Normal(jnp.zeros(X.shape[-1]), 1).to_event())
    logits = numpyro.deterministic("logits", (coef[..., None, :] @ X.T)[..., 0, :])
    with numpyro.plate("N", len(X)):
        return numpyro.sample("obs", dist.Bernoulli(logits=logits), obs=Y)


def fit_flow(model, num_samples=100 * 1000, seed=0):
    num_steps = 300 * 1000
    optimizer = optax.chain(optax.scale_by_adam(), optax.scale(-1.0),
        optax.scale_by_schedule(piecewise_constant_schedule(1.0e-3, {100 * 1000: 1.0e-4, 200 * 1000: 1.0e-5})))

    guide = AutoBNAFNormal(model, num_flows=1, hidden_factors=[8, 8], init_loc_fn=init_loc_fn)
    svi = SVI(model, guide, optimizer, Trace_ELBO(num_particles=1))
    svi_state = svi.init(random.PRNGKey(seed))
    svi_state = fori_loop(0, num_steps, partial(svi_body_fn, svi), svi_state)
    params = svi.get_params(svi_state)
    final_elbo = -Trace_ELBO(num_particles=num_samples).loss(random.PRNGKey(seed + 1), params, model, guide).item()

    return final_elbo


def fit_mean_field(model, num_samples=100 * 1000, diagonal=True, seed=0, iwae=False, K=4):
    num_steps = 900 * 1000
    sched = {300 * 1000: 1.0e-4, 600 * 1000: 1.0e-5}
    optimizer = optax.chain(optax.scale_by_adam(), optax.scale(-1.0),
                            optax.scale_by_schedule(piecewise_constant_schedule(1.0e-3, sched)))

    if diagonal:
        guide = AutoDiagonalNormal(model, init_scale=0.1, init_loc_fn=init_loc_fn)
    else:
        guide = AutoMultivariateNormal(model, init_scale=0.1, init_loc_fn=init_loc_fn)

    if iwae:
        svi = SVI(model, guide, optimizer, RenyiELBO(alpha=0, num_particles=K))
    else:
        svi = SVI(model, guide, optimizer, Trace_ELBO(num_particles=1))
    svi_state = svi.init(random.PRNGKey(seed))
    svi_state = fori_loop(0, num_steps, partial(svi_body_fn, svi), svi_state)
    params = svi.get_params(svi_state)

    if iwae:
        final_elbo = jnp.mean(lax.map(lambda key: -RenyiELBO(alpha=0, num_particles=K).loss(key, params, model, guide),
                                      random.split(random.PRNGKey(seed + 1), num_samples // K)))
    else:
        final_elbo = -Trace_ELBO(num_particles=num_samples).loss(random.PRNGKey(seed + 1), params, model, guide).item()

    return final_elbo, params, guide


def fit_dais(model, num_samples=100 * 1000, K=4, mf_params=None, seed=0):
    num_steps = 900 * 100000
    optimizer = optax.chain(optax.scale_by_adam(), optax.scale(-1.0),
        optax.scale_by_schedule(piecewise_constant_schedule(1.0e-4, {300 * 1000: 1.0e-5, 600 * 1000: 1.0e-6})))

    guide = AutoDAIS(model, K=K, init_loc_fn=init_loc_fn, eta_max=0.5, base_dist='diagonal', eta_init=0.005)
    init_params = {'auto_z_0_loc': mf_params['auto_loc'], 'auto_z_0_scale':  mf_params['auto_scale']}
    svi = SVI(model, guide, optimizer, Trace_ELBO(num_particles=1))
    svi_state = svi.init(random.PRNGKey(seed), init_params=init_params)

    def body_fn(i, val):
        svi_state, loss = svi.update(val)
        return svi_state

    svi_state = fori_loop(0, num_steps, body_fn, svi_state)
    params = svi.get_params(svi_state)
    final_elbo = -Trace_ELBO(num_particles=num_samples).loss(random.PRNGKey(seed + 1), params, model, guide).item()

    return final_elbo


def fit_rvrs(model, num_samples=100 * 1000, epsilon=1.0e-4, Z_target=0.5, seed=0,
             T_init=0.0, mf_params=None, mf_guide=None, diagonal=False):

    num_steps = 900 * 1000
    optimizer = optax.chain(optax.scale_by_adam(), optax.scale(-1.0),
        optax.scale_by_schedule(piecewise_constant_schedule(1.0e-4, {300 * 1000: 1.0e-5, 600 * 1000: 1.0e-6})))

    guide = AutoRVRS(model, S=2, T=T_init, guide=mf_guide, epsilon=epsilon, init_scale=np.nan, init_loc_fn=init_loc_fn,
                     adaptation_scheme="Z_target", T_lr=1.0, Z_target=Z_target, gamma=0.0)

    svi = SVI(model, guide, optimizer, Trace_ELBO())
    svi_state = svi.init(random.PRNGKey(seed), init_params=mf_params)
    svi_state = fori_loop(0, num_steps, partial(svi_body_fn, svi), svi_state)
    params = svi.get_params(svi_state)

    final_T = svi_state.mutable_state['_T_adapt']['value'].item()
    eval_guide = AutoRVRS(model, S=2, T=final_T, guide=mf_guide, epsilon=epsilon, init_scale=0.5, init_loc_fn=init_loc_fn,
                          adaptation_scheme='fixed', T_lr=0.0, include_log_Z=False)

    with numpyro.handlers.seed(rng_seed=0):
        eval_guide()

    predictive = Predictive(eval_guide, params=params, num_samples=num_samples // 2)
    posterior_samples = predictive(random.PRNGKey(seed + 1))
    a = np.ndarray.flatten(np.exp(posterior_samples['first_log_a']))
    log_Z = jnp.log(jnp.mean(a))

    final_elbo = jnp.mean(lax.map(lambda key: -Trace_ELBO().loss(key, params, model, eval_guide, multi_sample_guide=True),
                                  random.split(random.PRNGKey(seed + 2), num_samples // 2))) + log_Z

    stats = {'a': a, 'log_Z': log_Z.item(), 'Z': np.exp(log_Z).item(), 'T': final_T}

    return final_elbo, stats


def main(args):
    # ingest data and instantiate model
    XY = pd.read_csv('bank.csv', index_col=0)
    num_data = 100
    idx = np.random.RandomState(0).permutation(len(XY))[:num_data]
    X, Y = XY.values[idx, :-1], XY.values[idx, -1]
    model = partial(logistic_model, X, Y)

    print("Beginning training of a mix of variational methods on data from bank.csv...")

    # mean field variational inference
    mf_elbo, mf_params, mf_guide = fit_mean_field(model, diagonal=True, seed=args.seed)
    print("Mean Field ELBO:     {:.4f}".format(mf_elbo))
    results = {'MF': {'ELBO': mf_elbo}}

    for K in [8]:
        # iwae variational inference
        iwae_elbo, _, _ = fit_mean_field(model, diagonal=True, seed=args.seed, K=K, iwae=True)
        print("IWAE{} ELBO:     {:.4f}".format(K, iwae_elbo))
        results['IWAE{}'.format(K)] = {'ELBO': iwae_elbo}

    for Z_target in [0.3, 0.1]:
        # rvrs variational inference
        scheme_name = 'Z_target={:.2f}'.format(Z_target)
        rvrs_elbo, rvrs_stats = fit_rvrs(model, T_init=-mf_elbo, seed=args.seed, mf_params=mf_params,
                                         mf_guide=mf_guide, Z_target=Z_target, diagonal=True)
        result = {'Z_target': Z_target, 'T': rvrs_stats['T'], 'Z': rvrs_stats['Z'], 'ELBO': rvrs_elbo}

        s = "RVRS({}) ELBO: {:.4f}  Z_final: {:.3f}"
        print(s.format(scheme_name, rvrs_elbo, rvrs_stats['Z']))
        results['RVRS{}'.format(int(100 * Z_target))] = result

    # we don't run these methods by default since they can be slow (especially the flow)
    if args.include_all:
        for K in [4]:
            # dais/uha variational inference
            dais_elbo = fit_dais(model, K=K, mf_params=mf_params, seed=args.seed)
            results['DAIS{}'.format(K)] = {'ELBO': dais_elbo}
            s = "DAIS{}  ELBO: {:.4f}"
            print(s.format(K, dais_elbo))

        # variational inference with a full rank multivariate normal variational distribution
        mvn_chol_elbo, _, _ = fit_mean_field(model, diagonal=False, seed=args.seed)
        print("MultivariateNormal Cholesky ELBO:     {:.4f}".format(mvn_chol_elbo))
        results['MVNChol'] = {'ELBO': mvn_chol_elbo}

        # variational inference with a normalizing flow
        flow_elbo = fit_flow(model, seed=args.seed)
        print("Flow ELBO:     {:.4f}".format(flow_elbo))
        results['Flow'] = {'ELBO': flow_elbo}

    print("\n*** SUMMARY OF RESULTS ***\n")

    for method, result in results.items():
        if method == 'MF':
            continue
        s = "[{}]  DELTA ELBO: {:.3f}    (w.r.t. mean-field baseline)"
        print(s.format(method, result['ELBO'] - results['MF']['ELBO']))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='vary experiment')
    parser.add_argument('--seed', type=int, default=1, help="use to set random number seed")
    parser.add_argument('--include-all', action="store_true", default=False,
        help="by default we compare rvrs to mean-field and IWAE. use this option to expand the set of comparisons.")
    parser.add_argument("--device", default="cpu", type=str, choices =['cpu', 'gpu'],
        help="use `--device gpu` to run on CUDA. requires a GPU-enabled JAX installation.")
    parser.add_argument("--precision", default="single", type=str, choices=['single', 'double'],
        help="toggle between 32-bit and 64-bit numerical precision.")
    args = parser.parse_args()

    if args.precision == 'double':
        enable_x64()

    numpyro.set_platform(args.device)

    main(args)
