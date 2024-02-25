from collections import namedtuple
from functools import partial
import warnings

import numpy as np

import jax
from jax import lax, random, vmap
from jax.nn import log_sigmoid, sigmoid
import jax.numpy as jnp
from jax.lax import stop_gradient, select
from jax.scipy.special import logsumexp
from jax.tree_util import tree_map

import numpyro
from numpyro import handlers
import numpyro.distributions as dist
from numpyro.distributions import constraints
from numpyro.distributions.transforms import UnpackTransform, biject_to
from numpyro.distributions.util import sum_rightmost
from numpyro.handlers import replay, seed, trace
from numpyro.infer.autoguide import AutoContinuous, AutoDiagonalNormal, AutoGuide
from numpyro.infer.autoguide import _ravel_dict, _subsample_model, _unravel_dict
from numpyro.infer.elbo import Trace_ELBO as OrigTrace_ELBO
from numpyro.infer.hmc_util import dual_averaging
from numpyro.infer.initialization import init_to_uniform
from numpyro.infer.util import helpful_support_errors, log_density, transform_fn
from numpyro.infer.svi import SVI as OrigSVI
from numpyro.infer.svi import _make_loss_fn, SVIState
from numpyro.primitives import Messenger
from numpyro.util import _validate_model, check_model_guide_match, find_stack_level


class substitute(Messenger):

    def __init__(self, fn=None, data=None, substitute_fn=None):
        self.substitute_fn = substitute_fn
        self.data = data
        if sum((x is not None for x in (data, substitute_fn))) != 1:
            raise ValueError(
                "Only one of `data` or `substitute_fn` " "should be provided."
            )
        super(substitute, self).__init__(fn)

    def process_message(self, msg):
        if (msg["type"] not in ("sample", "param", "mutable", "plate", "deterministic")) or msg.get(
            "_control_flow_done", False
        ):
            if msg["type"] == "control_flow":
                if self.data is not None:
                    msg["kwargs"]["substitute_stack"].append(("substitute", self.data))
                if self.substitute_fn is not None:
                    msg["kwargs"]["substitute_stack"].append(
                        ("substitute", self.substitute_fn)
                    )
            return

        if self.data is not None:
            value = self.data.get(msg["name"])
        else:
            value = self.substitute_fn(msg)

        if value is not None:
            msg["value"] = value


class SVI(OrigSVI):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # TODO: use a better attribute for multi sample guide.
        self.multi_sample_guide = True if hasattr(self.guide, "S") else False

    def init(self, rng_key, *args, init_params=None, **kwargs):
        rng_key, model_seed, guide_seed = random.split(rng_key, 3)
        model_init = seed(self.model, model_seed)
        guide_init = seed(self.guide, guide_seed)
        if init_params is not None:
            guide_init = substitute(guide_init, init_params)
        guide_trace = trace(guide_init).get_trace(*args, **kwargs, **self.static_kwargs)
        init_guide_params = {
            name: site["value"]
            for name, site in guide_trace.items()
            if site["type"] == "param"
        }
        if init_params is not None:
            init_guide_params.update(init_params)
        if self.multi_sample_guide:
            latents = {
                name: site["value"][0]
                for name, site in guide_trace.items()
                if site["type"] == "sample" and site["value"].size > 0
            }
            latents.update(init_guide_params)
            with trace() as model_trace, substitute(data=latents):
                model_init(*args, **kwargs, **self.static_kwargs)
            for site in model_trace.values():
                if site["type"] == "mutable":
                    raise ValueError(
                        "mutable state in model is not supported for "
                        "multi-sample guide."
                    )
        else:
            model_trace = trace(
                substitute(replay(model_init, guide_trace), init_guide_params)
            ).get_trace(*args, **kwargs, **self.static_kwargs)

        params = {}
        inv_transforms = {}
        mutable_state = {}
        # NB: params in model_trace will be overwritten by params in guide_trace
        for site in list(model_trace.values()) + list(guide_trace.values()):
            if site["type"] == "param":
                constraint = site["kwargs"].pop("constraint", constraints.real)
                with helpful_support_errors(site):
                    transform = biject_to(constraint)
                inv_transforms[site["name"]] = transform
                params[site["name"]] = transform.inv(site["value"])
            elif site["type"] == "mutable":
                mutable_state[site["name"]] = site["value"]
            elif (
                site["type"] == "sample"
                and (not site["is_observed"])
                and site["fn"].support.is_discrete
                and not self.loss.can_infer_discrete
            ):
                s_name = type(self.loss).__name__
                warnings.warn(
                    f"Currently, SVI with {s_name} loss does not support models with discrete latent variables",
                    stacklevel=find_stack_level(),
                )

        if not mutable_state:
            mutable_state = None
        self.constrain_fn = partial(transform_fn, inv_transforms)
        # we convert weak types like float to float32/float64
        # to avoid recompiling body_fn in svi.run
        params, mutable_state = tree_map(
            lambda x: lax.convert_element_type(x, jnp.result_type(x)),
            (params, mutable_state),
        )
        return SVIState(self.optim.init(params), mutable_state, rng_key)

    def update(self, svi_state, *args, **kwargs):
        rng_key, rng_key_step = random.split(svi_state.rng_key)
        static_kwargs = self.static_kwargs.copy()
        if self.multi_sample_guide:
            static_kwargs["multi_sample_guide"] = True
        loss_fn = _make_loss_fn(
            self.loss,
            rng_key_step,
            self.constrain_fn,
            self.model,
            self.guide,
            args,
            kwargs,
            static_kwargs,
            mutable_state=svi_state.mutable_state,
        )
        (loss_val, mutable_state), optim_state = self.optim.eval_and_update(
            loss_fn, svi_state.optim_state
        )
        return SVIState(optim_state, mutable_state, rng_key), loss_val

    def stable_update(self, svi_state, *args, **kwargs):
        rng_key, rng_key_step = random.split(svi_state.rng_key)
        static_kwargs = self.static_kwargs.copy()
        if self.multi_sample_guide:
            static_kwargs["multi_sample_guide"] = True
        loss_fn = _make_loss_fn(
            self.loss,
            rng_key_step,
            self.constrain_fn,
            self.model,
            self.guide,
            args,
            kwargs,
            static_kwargs,
            mutable_state=svi_state.mutable_state,
        )
        (loss_val, mutable_state), optim_state = self.optim.eval_and_stable_update(
            loss_fn, svi_state.optim_state
        )
        return SVIState(optim_state, mutable_state, rng_key), loss_val

    def evaluate(self, svi_state, *args, **kwargs):
        # we split to have the same seed as `update_fn` given an svi_state
        _, rng_key_eval = random.split(svi_state.rng_key)
        params = self.get_params(svi_state)
        static_kwargs = self.static_kwargs.copy()
        if self.multi_sample_guide:
            static_kwargs["multi_sample_guide"] = True
        return self.loss.loss(
            rng_key_eval,
            params,
            self.model,
            self.guide,
            *args,
            **kwargs,
            **static_kwargs,
        )


class Trace_ELBO(OrigTrace_ELBO):

    def loss_with_mutable_state(
        self,
        rng_key,
        param_map,
        model,
        guide,
        *args,
        multi_sample_guide=False,
        **kwargs,
    ):
        def single_particle_elbo(rng_key):
            params = param_map.copy()
            model_seed, guide_seed = random.split(rng_key)
            seeded_guide = seed(guide, guide_seed)
            guide_log_density, guide_trace = log_density(
                seeded_guide, args, kwargs, param_map
            )
            mutable_params = {
                name: site["value"]
                for name, site in guide_trace.items()
                if site["type"] == "mutable"
            }
            params.update(mutable_params)
            if multi_sample_guide:
                plates = {
                    name: site["value"]
                    for name, site in guide_trace.items()
                    if site["type"] == "plate"
                }

                def get_model_density(key, latent):
                    with seed(rng_seed=key), substitute(data={**latent, **plates}):
                        model_log_density, _ = log_density(model, args, kwargs, params)
                    return model_log_density

                seeds = random.split(
                    model_seed, guide.S
                )  # todo: change the attribute name
                latents = {
                    name: site["value"]
                    for name, site in guide_trace.items()
                    if (site["type"] == "sample" and site["value"].size > 0)
                    or (site["type"] == "deterministic")
                }
                model_log_density = vmap(get_model_density)(seeds, latents)
                assert model_log_density.ndim == 1
                model_log_density = model_log_density.sum(0)
                # log p(z) - log q(z)
                elbo_particle = (model_log_density - guide_log_density) / seeds.shape[0]
            else:
                seeded_model = seed(model, model_seed)
                seeded_model = replay(seeded_model, guide_trace)
                model_log_density, model_trace = log_density(
                    seeded_model, args, kwargs, params
                )
                check_model_guide_match(model_trace, guide_trace)
                _validate_model(model_trace, plate_warning="loose")
                mutable_params.update(
                    {
                        name: site["value"]
                        for name, site in model_trace.items()
                        if site["type"] == "mutable"
                    }
                )
                # log p(z) - log q(z)
                elbo_particle = model_log_density - guide_log_density

            if mutable_params:
                if self.num_particles == 1:
                    return elbo_particle, mutable_params
                else:
                    return elbo_particle, None
                    raise ValueError(
                        "Currently, we only support mutable states with num_particles=1."
                    )
            else:
                return elbo_particle, None

        # Return (-elbo) since by convention we do gradient descent on a loss and
        # the ELBO is a lower bound that needs to be maximized.
        if self.num_particles == 1:
            elbo, mutable_state = single_particle_elbo(rng_key)
            return {"loss": -elbo, "mutable_state": mutable_state}
        else:
            rng_keys = random.split(rng_key, self.num_particles)
            if self.vectorize_particles:
                elbos, mutable_state = vmap(single_particle_elbo)(rng_keys)
            else:
                elbos, mutable_state = jax.lax.map(single_particle_elbo, rng_keys)
            return {"loss": -jnp.mean(elbos), "mutable_state": mutable_state}


TempAdaptState = namedtuple("TempAdaptState", ["temperature", "da_state", "window_idx"])


def temperature_adapter(
    init_temperature,
    num_adapt_steps=float("inf"),
    # find_reasonable_temperature=None,
    target_accept_prob=0.33,
):
    """
    A scheme to adapt tunable temperature during the warmup phase of HMC.
    :param int num_adapt_steps: Number of warmup steps.
    :param find_reasonable_temperature: A callable to find a reasonable temperature
        at the beginning of each adaptation window.
    :param float target_accept_prob: Target acceptance probability for temperature
        adaptation using Dual Averaging. Increasing this value will lead to a smaller
        temperature. Default to 0.33.
    :return: a pair of (`init_fn`, `update_fn`).
    """
    da_init, da_update = dual_averaging()
    init_window_size = 25

    def init_fn(temperature=-1.0):
        """
        :param float temperature: Initial temperature.
        :return: initial state of the adapt scheme.
        """
        da_state = da_init(-temperature)
        window_idx = jnp.array(0, dtype=jnp.result_type(int))
        return TempAdaptState(temperature, da_state, window_idx)

    def _update_at_window_end(state):
        temperature, da_state, window_idx = state
        da_state = da_init(-temperature)
        return TempAdaptState(temperature, da_state, window_idx + 1)

    def update_fn(t, accept_prob, state):
        """
        :param int t: The current time step.
        :param float accept_prob: Acceptance probability of the current time step.
        :param state: Current state of the adapt scheme.
        :return: new state of the adapt scheme.
        """
        temperature, da_state, window_idx = state
        da_state = da_update(target_accept_prob - accept_prob, da_state)
        # note: at the end of warmup phase, use average of log temperature
        neg_temperature, neg_temperature_avg, *_ = da_state
        temperature = jnp.where(
            t == (num_adapt_steps - 1), -neg_temperature_avg, -neg_temperature
        )
        t_at_window_end = t == (init_window_size * (2 ** (window_idx + 1) - 1) - 1)
        window_idx = jnp.where(t_at_window_end, window_idx + 1, window_idx)
        da_state = jax.lax.cond(
            t_at_window_end, -temperature, da_init, da_state, lambda x: x
        )
        return TempAdaptState(temperature, da_state, window_idx)

    return init_fn(init_temperature), update_fn


class AutoRVRS(AutoContinuous):
    """ """

    def __init__(
        self,
        model,
        *,
        S=4,  # number of samples
        T=0.0,
        T_lr=1.0,
        adaptation_scheme="Z_target",
        epsilon=0.1,
        guide=None,
        prefix="auto",
        init_loc_fn=init_to_uniform,
        init_scale=1.0,
        Z_target=0.33,
        T_exponent=None,
        gamma=0.99,  # controls momentum (0.0 => no momentum)
        num_warmup=float("inf"),
        include_log_Z=True,
        reparameterized=True,
        T_lr_drop=None,
    ):
        if S < 1:
            raise ValueError("S must satisfy S >= 1 (got S = {})".format(S))
        if init_scale <= 0.0:
            raise ValueError("init_scale must be positive.")
        # NOTE: removed because not jittable
        # if T is not None and not isinstance(T, float):
        #    raise ValueError("T must be None or a float.")
        if adaptation_scheme not in ["fixed", "Z_target", "dual_averaging"]:
            raise ValueError(
                "adaptation_scheme must be one of 'fixed', 'Z_target', or 'dual_averaging'."
            )

        self.S = S
        self.T = T
        self.epsilon = epsilon
        self.lambd = epsilon / (1 - epsilon)
        self.gamma = gamma
        self.include_log_Z = include_log_Z
        self.reparameterized = reparameterized
        self.T_lr_drop = T_lr_drop

        if guide is not None:
            if not isinstance(guide, AutoContinuous):
                raise ValueError("We only support AutoContinuous guide in AutoRVRS.")
            self.guide = guide
        else:
            self.guide = AutoDiagonalNormal(
                model, init_loc_fn=init_loc_fn, init_scale=init_scale, prefix=prefix
            )

        self.adaptation_scheme = adaptation_scheme
        self.T_lr = T_lr
        self.T_exponent = T_exponent
        self.Z_target = Z_target
        self.num_warmup = num_warmup
        super().__init__(model, prefix=prefix, init_loc_fn=init_loc_fn)

    def _setup_prototype(self, *args, **kwargs):
        super()._setup_prototype(*args, **kwargs)

        for name, site in self.prototype_trace.items():
            if (
                site["type"] == "plate"
                and isinstance(site["args"][1], int)
                and site["args"][0] > site["args"][1]
            ):
                raise NotImplementedError(
                    "AutoRVRS cannot be used in conjunction with data subsampling."
                )
        with handlers.block(), handlers.trace() as tr, handlers.seed(rng_seed=0):
            self.guide(*args, **kwargs)
        self.prototype_guide_trace = tr
        if self.adaptation_scheme == "dual_averaging":
            self._da_init_state, self._da_update = temperature_adapter(
                self.T, self.num_warmup, self.Z_target
            )

    def _get_posterior(self):
        raise NotImplementedError

    def _sample_latent(self, *args, **kwargs):
        model_params = {}
        for name, site in self.prototype_trace.items():
            if site["type"] == "param":
                model_params[name] = numpyro.param(
                    name,
                    site["value"],
                    constraint=site["kwargs"].pop("constraint", constraints.real),
                )
        guide_params = {}
        for name, site in self.prototype_guide_trace.items():
            if site["type"] == "param":
                guide_params[name] = numpyro.param(
                    name,
                    site["value"],
                    constraint=site["kwargs"].pop("constraint", constraints.real),
                )
        params = {"guide": guide_params, "model": model_params}

        def guide_sampler(key, params):
            with handlers.block(), handlers.seed(rng_seed=key), handlers.substitute(
                data=params["guide"]
            ):
                z = self.guide._sample_latent(*args, **kwargs)
            return z

        def guide_log_prob(z, params):
            z_and_params = {f"_{self.prefix}_latent": z, **params}
            with handlers.block():
                return log_density(
                    self.guide._sample_latent, args, kwargs, z_and_params
                )[0]

        def model_log_density(x, params):
            x_unpack = self._unpack_latent(x)
            with numpyro.handlers.block(), handlers.substitute(data=params):
                return -self._potential_fn(x_unpack)

        if self.adaptation_scheme == "Z_target":
            T_adapt = numpyro.primitives.mutable(
                "_T_adapt", {"value": jnp.array(self.T)}
            )
            if self.gamma != 0.0:
                T_grad_smoothed = numpyro.primitives.mutable(
                    "_T_grad_smoothed", {"value": jnp.array(0.0)}
                )
            T = T_adapt["value"]
        elif self.adaptation_scheme == "dual_averaging":
            T_adapt = numpyro.primitives.mutable(
                "_T_adapt", {"value": self._da_init_state}
            )
            T = T_adapt["value"].temperature
        else:
            T = self.T

        num_updates = numpyro.primitives.mutable(
            "_num_updates", {"value": jnp.array(0)}
        )

        def accept_log_prob_fn(z, params):
            params = jax.lax.stop_gradient(params)
            guide_lp = guide_log_prob(z, params["guide"])
            lw = model_log_density(z, params["model"]) - guide_lp
            a = sigmoid(lw + T)
            return jnp.log(self.epsilon + (1 - self.epsilon) * a), lw, guide_lp

        keys = random.split(numpyro.prng_key(), self.S)
        zs, log_weight, log_Z, first_log_a, guide_lp = batch_rejection_sampler_custom(
            accept_log_prob_fn, guide_sampler, keys, params
        )
        assert zs.shape == (self.S, self.latent_dim)

        numpyro.deterministic("first_log_a", first_log_a)

        # compute surrogate elbo
        az = sigmoid(log_weight + T)
        log_a_eps_z = jnp.log(self.epsilon + (1 - self.epsilon) * az)
        Az = log_weight - log_sigmoid(log_weight + T)
        A_bar = stop_gradient(Az - Az.mean(0))
        assert az.shape == Az.shape == log_weight.shape == (self.S,)

        S_ratio = self.S / (self.S - 1)

        if self.reparameterized:
            ratio = (self.lambd + jnp.square(az)) / (self.lambd + az)
            ratio_bar = stop_gradient(ratio)
            surrogate = (
                S_ratio * (A_bar * (ratio_bar * log_a_eps_z + ratio)).sum()
                + (ratio_bar * Az).sum()
            )
            # surrogate = S_ratio * (2 * A_bar * az).sum() + (stop_gradient(az) * Az).sum()
        else:
            # recompute for simplicity
            guide_lp = jax.vmap(
                lambda z: guide_log_prob(stop_gradient(z), params["guide"])
            )(zs)
            assert guide_lp.shape == (self.S,)
            surrogate = (
                S_ratio
                * (
                    stop_gradient(A_bar * (self.epsilon + (1 - self.epsilon) * az))
                    * guide_lp
                ).sum()
            )

        if self.include_log_Z:
            elbo_correction = stop_gradient(
                surrogate + guide_lp.sum() + log_a_eps_z.sum() - log_Z * self.S
            )
        else:
            elbo_correction = stop_gradient(
                surrogate + guide_lp.sum() + log_a_eps_z.sum()
            )
        numpyro.factor("surrogate_factor", -surrogate + elbo_correction)

        num_updates["value"] = num_updates["value"] + 1

        if self.adaptation_scheme == "Z_target":
            # minimize (Z - Z_target) ** 2
            a = stop_gradient(jnp.exp(first_log_a))
            a_minus = 1 / (self.S - 1) * (jnp.sum(a) - a)
            T_grad = jnp.mean((a_minus - self.Z_target) * a * (1 - a))

            if self.gamma != 0.0:
                T_grad_smoothed["value"] = (
                    self.gamma * T_grad_smoothed["value"] + (1.0 - self.gamma) * T_grad
                )
                T_grad = T_grad_smoothed["value"]

            if self.T_exponent is not None:
                T_lr = self.T_lr * jnp.power(num_updates["value"], -self.T_exponent)
            elif self.T_lr_drop is not None:
                T_lr = self.T_lr * 0.2 ** (
                    num_updates["value"] // self.T_lr_drop
                ).astype(float)
            else:
                T_lr = self.T_lr

            T_adapt["value"] = T_adapt["value"] - T_lr * T_grad
        elif self.adaptation_scheme == "dual_averaging":
            # Z = stop_gradient(jnp.exp(log_Z))
            # TODO: use log_a_sum instead of first_log_a?
            a = stop_gradient(jnp.exp(first_log_a))
            a_minus = 1 / (self.S - 1) * (jnp.sum(a) - a)
            T_grad = jnp.mean((a_minus - self.Z_target) * a * (1 - a))
            Z = self.Z_target + T_grad
            t = num_updates["value"]
            T_adapt["value"] = lax.cond(
                t < self.num_warmup,
                (t, Z, T_adapt["value"]),
                lambda args: self._da_update(*args),
                T_adapt["value"],
                lambda x: x,
            )
            # num_updates["value"] = t + 1

        return stop_gradient(zs)

    def __call__(self, *args, **kwargs):
        if self.prototype_trace is None:
            # run model to inspect the model structure
            self._setup_prototype(*args, **kwargs)

        latent = self._sample_latent(*args, **kwargs)

        # unpack continuous latent samples
        result = {}

        for name, unconstrained_value in jax.vmap(self._unpack_latent)(latent).items():
            site = self.prototype_trace[name]
            with helpful_support_errors(site):
                transform = biject_to(site["fn"].support)
            value = jax.vmap(transform)(unconstrained_value)
            event_ndim = site["fn"].event_dim
            if numpyro.get_mask() is False:
                log_density = 0.0
            else:
                log_density = -jax.vmap(transform.log_abs_det_jacobian)(
                    unconstrained_value, value
                )
                log_density = sum_rightmost(
                    log_density, jnp.ndim(log_density) - jnp.ndim(value) + event_ndim
                )
            delta_dist = dist.Delta(
                value, log_density=log_density, event_dim=event_ndim
            )
            result[name] = numpyro.sample(name, delta_dist)

        return result

    def sample_posterior(self, rng_key, params, sample_shape=()):
        def _single_sample(_rng_key):
            with handlers.trace() as tr:
                latent_sample = handlers.substitute(
                    handlers.seed(self._sample_latent, _rng_key), params
                )(sample_shape=())
            deterministics = {
                name: site["value"]
                for name, site in tr.items()
                if site["type"] == "deterministic"
            }
            latents = jax.vmap(self._unpack_and_constrain, in_axes=(0, None))(
                latent_sample, params
            )
            return {**deterministics, **latents}

        if sample_shape:
            rng_key = random.split(rng_key, int(np.prod(sample_shape)))
            samples = lax.map(_single_sample, rng_key)
            return tree_map(
                lambda x: jnp.reshape(x, sample_shape + jnp.shape(x)[1:]),
                samples,
            )
        else:
            return _single_sample(rng_key)


def rejection_sampler(accept_log_prob_fn, guide_sampler, key):
    def cond_fn(val):
        return ~val[-1]

    def body_fn(val):
        key, _, _, log_a_sum, num_samples, first_log_a, _, _ = val
        key_next, key_uniform, key_q = random.split(key, 3)
        z = guide_sampler(key_q)
        accept_log_prob, log_weight, guide_lp = accept_log_prob_fn(z)
        log_u = -random.exponential(key_uniform)
        is_accepted = log_u < accept_log_prob
        first_log_a = select(num_samples == 0, accept_log_prob, first_log_a)
        log_a_sum = logsumexp(jnp.stack([log_a_sum, accept_log_prob]))
        return (
            key_next,
            z,
            log_weight,
            log_a_sum,
            num_samples + 1,
            first_log_a,
            guide_lp,
            is_accepted,
        )

    prototype_z = tree_map(
        lambda x: jnp.zeros(x.shape, dtype=x.dtype), jax.eval_shape(guide_sampler, key)
    )
    init_val = (key, prototype_z, -jnp.inf, -jnp.inf, 0, -jnp.inf, 0, False)
    _, z, log_w, log_a_sum, num_samples, first_log_a, guide_lp, _ = jax.lax.while_loop(
        cond_fn, body_fn, init_val
    )

    return z, log_w, log_a_sum, num_samples, first_log_a, guide_lp


def batch_rejection_sampler(accept_log_prob_fn, guide_sampler, keys, params):
    def sample_and_accept_fn(key, params):
        z = guide_sampler(key, params)
        return z, accept_log_prob_fn(z, params)

    z_init = tree_map(
        lambda x: jnp.zeros(x.shape, dtype=x.dtype),
        jax.eval_shape(guide_sampler, keys[0], params),
    )
    return _rs_impl(sample_and_accept_fn, z_init, keys, params)[1:]


def _rs_impl(sample_and_accept_fn, z_init, keys, params):
    assert keys.ndim == 2
    S = keys.shape[0]
    zs_init = tree_map(lambda x: jnp.broadcast_to(x, (S,) + jnp.shape(x)), z_init)
    neg_inf = jnp.full(S, -jnp.inf)
    buffer = (keys, zs_init, neg_inf, neg_inf, jnp.full(S, False))
    init_val = (keys, neg_inf, neg_inf, jnp.array(0), buffer)

    def cond_fn(val):
        is_accepted = val[-1][-1]
        return is_accepted.sum() < S

    def body_fn(key, params):
        key_next, key_uniform, key_q = random.split(key, 3)
        z, (accept_log_prob, log_weight, guide_lp) = sample_and_accept_fn(key_q, params)
        log_u = -random.exponential(key_uniform)
        is_accepted = log_u < accept_log_prob
        return key_next, accept_log_prob, (key_q, z, log_weight, guide_lp, is_accepted)

    def batch_body_fn(val):
        keys, log_a_sum, first_log_a, num_samples, buffer = val
        keys_next, accept_log_prob, candidate = jax.vmap(body_fn, in_axes=(0, None))(
            keys, params
        )
        log_a_sum = logsumexp(jnp.stack([log_a_sum, accept_log_prob], axis=0), axis=0)
        buffer_extend = tree_map(
            lambda a, b: jnp.concatenate([a, b]), candidate, buffer
        )
        is_accepted = buffer_extend[-1]
        maybe_accept_indices = jnp.argsort(is_accepted)[-S:]
        new_buffer = tree_map(lambda x: x[maybe_accept_indices], buffer_extend)
        first_log_a = select(num_samples == 0, accept_log_prob, first_log_a)
        return keys_next, log_a_sum, first_log_a, num_samples + 1, new_buffer

    _, log_a_sum, first_log_a, num_samples, buffer = jax.lax.while_loop(
        cond_fn, batch_body_fn, init_val
    )
    key_q, z, log_w, guide_lp, _ = buffer
    log_Z = logsumexp(log_a_sum) - jnp.log(num_samples * S)
    return key_q, z, log_w, log_Z, first_log_a, guide_lp


def batch_rejection_sampler_custom(accept_log_prob_fn, guide_sampler, keys, params):
    def sample_and_accept_fn(key, params):
        z = guide_sampler(key, params)
        return z, accept_log_prob_fn(z, params)

    z_init = tree_map(
        lambda x: jnp.zeros(x.shape, dtype=x.dtype),
        jax.eval_shape(guide_sampler, keys[0], params),
    )
    return _rs_custom_impl(sample_and_accept_fn, z_init, keys, params)[1:]


@partial(jax.custom_vjp, nondiff_argnums=(0,))
def _rs_custom_impl(sample_and_accept_fn, z_init, keys, params):
    return _rs_impl(sample_and_accept_fn, z_init, keys, params)


def _rs_fwd(sample_and_accept_fn, z_init, keys, params):
    out = _rs_custom_impl(sample_and_accept_fn, z_init, keys, params)
    key_q = out[0]
    return out, (key_q, params)


def _rs_bwd(sample_and_accept_fn, res, g):
    key_q, params = res
    _, z_grads, lw_grads, *_ = g

    def get_z_and_lw(key, params):
        z, (_, lw, _) = sample_and_accept_fn(key, params)
        return z, lw

    def sample_grad(key, z_grad, lw_grad):
        _, guide_vjp = jax.vjp(partial(get_z_and_lw, key), params)
        return guide_vjp((z_grad, lw_grad))[0]

    batch_params_grad = jax.vmap(sample_grad)(key_q, z_grads, lw_grads)
    params_grad = jax.tree_util.tree_map(lambda x: x.sum(0), batch_params_grad)
    return (None, None, params_grad)


_rs_custom_impl.defvjp(_rs_fwd, _rs_bwd)


class AutoSemiRVRS(AutoGuide):
    """
    This implementation of :class:`AutoSemiRVRS` [1] combines a parametric variational
    distribution over global latent variables with RVRS to infer local latent variables.
    Unlike :class:`AutoRVRS` this guide can be used in conjunction with data subsampling.
    Usage::
        def global_model():
            return numpyro.sample("theta", dist.Normal(0, 1))
        def local_model(theta):
            with numpyro.plate("data", 8, subsample_size=2):
                tau = numpyro.sample("tau", dist.Gamma(5.0, 5.0))
                numpyro.sample("obs", dist.Normal(0.0, tau), obs=jnp.ones(2))
        global_guide = AutoNormal(global_model)
        local_guide = AutoNormal(local_model)
        model = lambda: local_model(global_model())
        guide = AutoSemiRVRS(model, local_model, global_guide, local_guide)
        svi = SVI(model, guide, ...)
        # sample posterior for particular data subset {3, 7}
        with handlers.substitute(data={"data": jnp.array([3, 7])}):
            samples = guide.sample_posterior(random.PRNGKey(1), params)
    :param callable model: A NumPyro model with global and local latent variables.
    :param callable global_guide: A guide for the global latent variables, e.g. an autoguide.
        The return type should be a dictionary of latent sample sites names and corresponding samples.
    :param callable local_guide: An auto guide for the local latent variables.
    :param str prefix: A prefix that will be prefixed to all internal sites.
    """

    def __init__(
        self,
        model,
        local_model,
        global_guide,
        local_guide,
        *,
        prefix="auto",
        S=4,
        T=0.0,
        T_lr=1.0,
        adaptation_scheme="Z_target",
        epsilon=0.1,
        init_loc_fn=init_to_uniform,
        Z_target=0.33,
        T_exponent=None,
        gamma=0.99,  # controls momentum (0.0 => no momentum)
        num_warmup=float("inf"),
        include_log_Z=True,
        reparameterized=True,
        T_lr_drop=None,
        subsample_plate=None,
        max_rs_steps=None,
        relocate_resource_when_accepted=False,
    ):
        if S < 1:
            raise ValueError("S must satisfy S >= 1 (got S = {})".format(S))
        # if T is not None and not isinstance(T, float):
        #     raise ValueError("T must be None or a float.")
        if adaptation_scheme not in ["fixed", "Z_target", "dual_averaging"]:
            raise ValueError(
                "adaptation_scheme must be one of 'fixed', 'Z_target', or 'dual_averaging'."
            )

        self.local_model = local_model
        self.global_guide = global_guide
        self.local_guide = local_guide
        self.S = S
        self.T = T
        self.epsilon = epsilon
        self.lambd = epsilon / (1 - epsilon)
        self.gamma = gamma
        self.include_log_Z = include_log_Z
        self.reparameterized = reparameterized
        self.T_lr_drop = T_lr_drop
        self.adaptation_scheme = adaptation_scheme
        self.T_lr = T_lr
        self.T_exponent = T_exponent
        self.Z_target = Z_target
        self.num_warmup = num_warmup
        self.subsample_plate = subsample_plate
        self.max_rs_steps = max_rs_steps
        self.relocate_resource_when_accepted = relocate_resource_when_accepted
        super().__init__(model, prefix=prefix, init_loc_fn=init_loc_fn)

    def _setup_prototype(self, *args, **kwargs):
        super()._setup_prototype(*args, **kwargs)
        assert isinstance(self.prototype_trace, dict)
        # extract global/local/local_dim/plates
        subsample_plates = {
            name: site
            for name, site in self.prototype_trace.items()
            if site["type"] == "plate"
            and isinstance(site["args"][1], int)
            and site["args"][0] > site["args"][1]
        }
        if self.subsample_plate is not None:
            subsample_plates[self.subsample_plate] = self.prototype_trace[self.subsample_plate]
        num_plates = len(subsample_plates)
        assert (
            num_plates == 1
        ), f"AutoSemiRVRS assumes that the model contains exactly 1 plate with data subsampling but got {num_plates}."
        plate_name = list(subsample_plates.keys())[0]
        local_vars = []
        subsample_axes = {}
        plate_dim = None
        for name, site in self.prototype_trace.items():
            if site["type"] == "sample" and not site["is_observed"]:
                for frame in site["cond_indep_stack"]:
                    if frame.name == plate_name:
                        if plate_dim is None:
                            plate_dim = frame.dim
                        local_vars.append(name)
                        subsample_axes[name] = plate_dim - site["fn"].event_dim
                        break
        if len(local_vars) == 0:
            raise RuntimeError(
                "There are no local variables in the `{plate_name}` plate."
                " AutoSemiDAIS is appropriate for models with local variables."
            )

        local_init_locs = {
            name: site["value"]
            for name, site in self.prototype_trace.items()
            if name in local_vars
        }

        one_sample = {
            k: jnp.take(v, 0, axis=subsample_axes[k])
            for k, v in local_init_locs.items()
        }
        _, shape_dict = _ravel_dict(one_sample)
        self._pack_local_latent = jax.vmap(
            lambda x: _ravel_dict(x)[0], in_axes=(subsample_axes,)
        )
        local_init_latent = self._pack_local_latent(local_init_locs)
        unpack_latent = partial(_unravel_dict, shape_dict=shape_dict)
        # this is to match the behavior of Pyro, where we can apply
        # unpack_latent for a batch of samples
        self._unpack_local_latent = jax.vmap(
            UnpackTransform(unpack_latent), out_axes=subsample_axes
        )
        plate_full_size, plate_subsample_size = subsample_plates[plate_name]["args"]
        self._local_latent_dim = jnp.size(local_init_latent) // plate_subsample_size
        self._local_plate = (plate_name, plate_full_size, plate_subsample_size)

        if self.global_guide is not None:
            with handlers.block(), handlers.trace() as tr, handlers.seed(rng_seed=0):
                self.global_guide(*args, **kwargs)
            self.prototype_global_guide_trace = tr

            with handlers.block(), handlers.seed(rng_seed=0):
                local_args = (self.global_guide.model(*args, **kwargs),)
                local_kwargs = {}
        else:
            local_args = args
            local_kwargs = kwargs

        with handlers.block(), handlers.trace() as tr, handlers.seed(rng_seed=0):
            self.local_guide(*local_args, **local_kwargs)
        self.prototype_local_guide_trace = tr

        with handlers.block(), handlers.trace() as tr, handlers.seed(rng_seed=0):
            self.local_model(*local_args, **local_kwargs)
        self.prototype_local_model_trace = tr

        if self.adaptation_scheme == "dual_averaging":
            self._da_init_state, self._da_update = temperature_adapter(
                self.T, self.num_warmup, self.Z_target
            )

    def __call__(self, *args, **kwargs):
        if self.prototype_trace is None:
            # run model to inspect the model structure
            self._setup_prototype(*args, **kwargs)
        assert isinstance(self.prototype_trace, dict)

        global_latents, local_latent_flat = self._sample_latent(*args, **kwargs)

        # unpack continuous latent samples
        result = {}
        for name, value in global_latents.items():
            site = self.prototype_trace[name]
            event_ndim = site["fn"].event_dim
            delta_dist = dist.Delta(value, log_density=0.0, event_dim=event_ndim)
            result[name] = numpyro.sample(name, delta_dist)

        for name, value in jax.vmap(self._unpack_local_latent)(
            local_latent_flat
        ).items():
            site = self.prototype_trace[name]
            event_ndim = site["fn"].event_dim
            # Note: "surrogate_factor"'s guide_lp is log density in constrained space.
            delta_dist = dist.Delta(value, log_density=0.0, event_dim=event_ndim)
            result[name] = numpyro.sample(name, delta_dist)

        return result

    def _get_posterior(self):
        raise NotImplementedError

    def _sample_latent(self, *args, **kwargs):
        kwargs.pop("sample_shape", ())
        plate_name, N, M = self._local_plate
        subsample_plate = numpyro.plate(plate_name, N, subsample_size=M)
        subsample_idx = subsample_plate._indices
        M = subsample_idx.shape[0]

        model_params = {}
        assert isinstance(self.prototype_trace, dict)
        for name, site in self.prototype_trace.items():
            if site["type"] == "param":
                model_params[name] = numpyro.param(
                    name, site["value"], **site["kwargs"]
                )

        global_key = numpyro.prng_key()
        global_guide_params = {}
        global_lp = 0.0
        if self.global_guide is not None:
            for name, site in self.prototype_global_guide_trace.items():
                if site["type"] == "param":
                    global_guide_params[name] = numpyro.param(
                        name, site["value"], **site["kwargs"]
                    )
            with handlers.block(), handlers.trace() as tr, handlers.seed(
                rng_seed=global_key
            ), handlers.substitute(data=global_guide_params):
                self.global_guide(*args, **kwargs)
            global_latents = {
                name: site["value"]
                for name, site in tr.items()
                if site["type"] == "sample" and not site.get("is_observed", False)
            }
            for name, site in tr.items():
                if name in global_latents:
                    global_lp = global_lp + site["fn"].log_prob(site["value"]).sum()

            rng_key = numpyro.prng_key()
            with handlers.block(), handlers.seed(rng_seed=rng_key), handlers.substitute(
                data=dict(**global_latents, **model_params)
            ):
                local_args = (
                    jax.lax.stop_gradient(self.global_guide.model(*args, **kwargs)),
                )
                local_kwargs = {}
        else:
            local_args = args
            local_kwargs = kwargs
            global_latents = {}

        assert isinstance(self.prototype_local_guide_trace, dict)
        local_guide_params = {}
        for name, site in self.prototype_local_guide_trace.items():
            if site["type"] == "param":
                local_guide_params[name] = numpyro.param(
                    name, site["value"], **site["kwargs"]
                )

        subsample_model = partial(_subsample_model, self.local_model)
        subsample_guide = partial(_subsample_model, self.local_guide)

        def single_local_model_log_density(z, subsample_idx):
            latent = self._unpack_local_latent(z)
            with handlers.block():
                # Scale down potential_fn by N because potential_fn scales up the log_prob.
                # We skip the warning because we are using subsample_idx with size 1 for a
                # plate with subsample_size M.
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    kwargs = {"_subsample_idx": {plate_name: subsample_idx}}
                    scale = N / subsample_idx.shape[0]
                    kwargs.update(local_kwargs)
                    latent_and_params = dict(
                        **latent, **jax.lax.stop_gradient(model_params)
                    )
                    return (
                        log_density(
                            subsample_model, local_args, kwargs, latent_and_params
                        )[0]
                        / scale
                    )

        def local_model_log_density(z, subsample_idx):
            # shape: local_latent_flat -> (M,) | subsample_idx -> (M,) | out -> (M,)
            return jax.vmap(single_local_model_log_density)(
                jnp.expand_dims(z, 1), jnp.expand_dims(subsample_idx, 1)
            )

        def single_local_guide_sampler(subsample_idx, key, params):
            with handlers.block(), handlers.trace() as tr, handlers.seed(
                rng_seed=key
            ), handlers.substitute(data=params):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    kwargs = {"_subsample_idx": {plate_name: subsample_idx}}
                    kwargs.update(local_kwargs)
                    subsample_guide(*local_args, **kwargs)
            latent = {
                name: site["value"]
                for name, site in tr.items()
                if site["type"] == "sample" and not site.get("is_observed", False)
            }
            z = self._pack_local_latent(latent)
            return z  # flatten array in constrained space

        def local_guide_sampler(subsample_idx, key, params):
            # shape: params -> (N,) | key -> (M,) | subsample_idx -> (M,) | out -> (M,)
            return jax.vmap(single_local_guide_sampler, (0, 0, None))(
                jnp.expand_dims(subsample_idx, 1), key, params
            )[:, 0]

        def single_local_guide_log_density(z, subsample_idx, params):
            latent = self._unpack_local_latent(z)
            assert isinstance(latent, dict)
            with handlers.block():
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    kwargs = {"_subsample_idx": {plate_name: subsample_idx}}
                    scale = N / subsample_idx.shape[0]
                    kwargs.update(local_kwargs)
                    return (
                        log_density(
                            subsample_guide, local_args, kwargs, {**params, **latent}
                        )[0]
                        / scale
                    )

        def local_guide_log_density(z, subsample_idx, params):
            # shape: params -> (N,) | z -> (M,) | subsample_idx -> (M,) | out -> (M,)
            return jax.vmap(single_local_guide_log_density, (0, 0, None))(
                jnp.expand_dims(z, 1), jnp.expand_dims(subsample_idx, 1), params
            )

        if self.adaptation_scheme == "Z_target":
            T_adapt = numpyro.primitives.mutable(
                "_T_adapt", {"value": jnp.full(N, self.T)}
            )
            if self.gamma != 0.0:
                T_grad_smoothed = numpyro.primitives.mutable(
                    "_T_grad_smoothed", {"value": jnp.full(N, 0.0)}
                )
            num_updates = numpyro.primitives.mutable(
                "_num_updates", {"value": jnp.full(N, 0)}
            )
            T = T_adapt["value"]
        elif self.adaptation_scheme == "dual_averaging":
            init_value = tree_map(
                lambda x: jnp.broadcast_to(x, (N,) + jnp.shape(x)), self._da_init_state
            )
            T_adapt = numpyro.primitives.mutable("_T_adapt", {"value": init_value})
            num_updates = numpyro.primitives.mutable(
                "_num_updates", {"value": jnp.full(N, 0)}
            )
            T = T_adapt["value"].temperature
        else:
            T = jnp.full(N, self.T)

        def accept_log_prob_fn(z, subsample_idx, params):
            # shape: z -> (M,) | params -> (N,) | subsample_idx -> (M,) | out -> (M,)
            params = jax.lax.stop_gradient(params)
            guide_lp = local_guide_log_density(z, subsample_idx, params)
            lw = local_model_log_density(z, subsample_idx) - guide_lp
            a = sigmoid(lw + T[subsample_idx])
            return jnp.log(self.epsilon + (1 - self.epsilon) * a), lw, guide_lp

        rs_key, resample_key = random.split(numpyro.prng_key())
        zs, log_weight, log_Z, first_log_a, guide_lp, is_accepted = rs_local(
            accept_log_prob_fn,
            local_guide_sampler,
            rs_key,
            resample_key if (self.relocate_resource_when_accepted or self.max_rs_steps) else None,
            local_guide_params,
            subsample_idx,
            max_rs_steps=self.max_rs_steps,
            rs_particles=self.S,
        )
        assert T.shape == (N,)
        assert zs.shape == (self.S, M, self._local_latent_dim)
        assert log_weight.shape == (self.S, M)
        assert log_Z.shape == (M,)
        assert first_log_a.shape == (self.S, M)
        assert guide_lp.shape == (self.S, M)

        if not isinstance(self.include_log_Z, bool):
            assert isinstance(self.include_log_Z, int)
            keys_ = jax.random.split(numpyro.prng_key(), self.include_log_Z)
            zs_ = jax.vmap(single_local_guide_sampler, (None, 0, None))(subsample_idx, keys_, local_guide_params)
            assert zs_.shape == (self.include_log_Z, M, self._local_latent_dim)
            first_log_a, *_ = jax.vmap(accept_log_prob_fn, (0, None, None))(zs_, subsample_idx, local_guide_params)
            assert first_log_a.shape == (self.include_log_Z, M)
            log_Z = logsumexp(first_log_a, axis=0) - jnp.log(self.include_log_Z)
            assert log_Z.shape == (M,)
        else:
            numpyro.deterministic("first_log_a", first_log_a)

        # compute surrogate elbo
        log_weight_T = log_weight + T[subsample_idx]
        az = sigmoid(log_weight_T)
        log_a_eps_z = jnp.log(self.epsilon + (1 - self.epsilon) * az)
        Az = log_weight - log_sigmoid(log_weight_T)
        A_bar = stop_gradient(Az - Az.mean(0))
        assert az.shape == Az.shape == log_weight.shape == (self.S, M)

        assert self.reparameterized
        ratio = (self.lambd + jnp.square(az)) / (self.lambd + az)
        ratio_bar = stop_gradient(ratio)
        surrogate = (
            self.S / max(self.S - 1, 1) * (A_bar * (ratio_bar * log_a_eps_z + ratio)).sum(0)
            + (ratio_bar * Az).sum(0)
        )

        if self.include_log_Z:
            elbo_correction = stop_gradient(
                surrogate + guide_lp.sum(0) + log_a_eps_z.sum(0) - log_Z * self.S
            )
        else:
            elbo_correction = stop_gradient(
                surrogate + guide_lp.sum(0) + log_a_eps_z.sum(0)
            )
        # Scale the factor by subsample factor N / M
        mask = is_accepted.sum(0) == self.S
        assert mask.shape == (M,)
        mask_sum = jnp.maximum(mask.sum(0), 1.)
        factor = ((-surrogate + elbo_correction) * mask).sum(0) * N / mask_sum + global_lp * self.S
        numpyro.factor("surrogate_factor", factor)
        if self.max_rs_steps is not None:
            numpyro.deterministic("mask", jnp.broadcast_to(mask, (self.S, M)))

        if self.adaptation_scheme == "Z_target":
            # minimize (Z - Z_target) ** 2
            a = stop_gradient(jnp.exp(first_log_a))
            a_minus = 1 / max(self.S - 1, 1) * (a.sum(0) - a)
            T_grad = jnp.mean((a_minus - self.Z_target) * a * (1 - a), axis=0)
            assert T_grad.shape == (M,)

            num_updates["value"] = (
                num_updates["value"]
                .at[subsample_idx]
                .set(num_updates["value"][subsample_idx] + 1)
            )
            if self.gamma != 0.0:
                T_grad = (
                    self.gamma * T_grad_smoothed["value"][subsample_idx]
                    + (1.0 - self.gamma) * T_grad
                )
                T_grad_smoothed["value"] = (
                    T_grad_smoothed["value"].at[subsample_idx].set(T_grad)
                )

            if self.T_exponent is not None:
                T_lr = self.T_lr * jnp.power(
                    num_updates["value"][subsample_idx], -self.T_exponent
                )
            elif self.T_lr_drop is not None:
                T_lr = self.T_lr * 0.1 ** (
                    num_updates["value"][subsample_idx] // self.T_lr_drop
                ).astype(jnp.result_type(float))
            else:
                T_lr = self.T_lr

            T_adapt["value"] = (
                T_adapt["value"]
                .at[subsample_idx]
                .set(T_adapt["value"][subsample_idx] - T_lr * T_grad)
            )
        elif self.adaptation_scheme == "dual_averaging":
            a = stop_gradient(jnp.exp(first_log_a))
            a_minus = 1 / max(self.S - 1, 1) * (a.sum(0) - a)
            T_grad = jnp.mean((a_minus - self.Z_target) * a * (1 - a), axis=0)
            Z = self.Z_target + T_grad
            t = num_updates["value"][subsample_idx]
            T_adapt_old = tree_map(lambda x: x[subsample_idx], T_adapt["value"])
            assert self.num_warmup == float("inf")
            T_adapt_new = jax.vmap(self._da_update)(t, Z, T_adapt_old)
            T_adapt["value"] = tree_map(
                lambda x, x_new: x.at[subsample_idx].set(x_new),
                T_adapt["value"],
                T_adapt_new,
            )
            num_updates["value"] = num_updates["value"].at[subsample_idx].set(t + 1)

        global_latents = tree_map(
            lambda x: jnp.broadcast_to(x, (self.S,) + x.shape), global_latents
        )
        return global_latents, stop_gradient(zs)

    def sample_posterior(self, rng_key, params, *args, sample_shape=(), parallel=False, **kwargs):
        def _single_sample(_rng_key):
            with handlers.trace() as tr:
                global_latents, local_flat = handlers.substitute(
                    handlers.seed(self._sample_latent, _rng_key), params
                )(*args, **kwargs)
            deterministics = {
                name: site["value"]
                for name, site in tr.items()
                if site["type"] == "deterministic"
            }

            def unpack(global_latents, local_flat):
                local_latents = self._unpack_local_latent(local_flat)
                return {**global_latents, **local_latents}

            return dict(
                **deterministics, **jax.vmap(unpack)(global_latents, local_flat)
            )

        if sample_shape:
            rng_key = random.split(rng_key, int(np.prod(sample_shape)))
            if parallel:
                samples = jax.vmap(_single_sample)(rng_key)
            else:
                samples = lax.map(_single_sample, rng_key)
            return tree_map(
                lambda x: jnp.reshape(x, sample_shape + jnp.shape(x)[1:]), samples
            )
        else:
            return _single_sample(rng_key)


def get_systematic_resampling_indices(weights, rng_key, num_samples):
    """Gets resampling indices based on systematic resampling."""
    n = weights.shape[0]
    cummulative_weight = weights.cumsum(axis=0)
    cummulative_weight = cummulative_weight / cummulative_weight[-1]
    cummulative_weight = cummulative_weight.reshape((n, -1)).swapaxes(0, 1)
    m = cummulative_weight.shape[0]
    uniform = jax.random.uniform(rng_key, (m,))
    positions = (uniform[:, None] + np.arange(num_samples)) / num_samples
    shift = jnp.arange(m)[:, None]
    cummulative_weight = (cummulative_weight + 2 * shift).reshape(-1)
    positions = (positions + 2 * shift).reshape(-1)
    index = cummulative_weight.searchsorted(positions)
    index = (index.reshape(m, num_samples) - n * shift).swapaxes(0, 1)
    return index.reshape((num_samples,) + weights.shape[1:])


def rs_local(
    accept_log_prob_fn, guide_sampler, rs_key, resample_key, params, subsample_idx, max_rs_steps, rs_particles
):
    def sample_and_accept_fn(subsample_idx, key, params):
        z = guide_sampler(subsample_idx, key, params)
        return z, accept_log_prob_fn(z, subsample_idx, params)

    M = subsample_idx.shape[0]
    z_init = tree_map(
        lambda x: jnp.zeros(x.shape, dtype=x.dtype),
        jax.eval_shape(guide_sampler, subsample_idx, random.split(rs_key, M), params),
    )
    return _rs_local_custom_impl(
        sample_and_accept_fn, z_init, subsample_idx, rs_key, resample_key, params, max_rs_steps, rs_particles
    )[1:]


def _subsample_indexing(xs, idxs):
    return jax.vmap(lambda x, idx: x[idx], in_axes=1, out_axes=1)(xs, idxs)


def _rs_local_impl(
    sample_and_accept_fn, z_init, subsample_idx, rs_key, resample_key, params, max_rs_steps, rs_particles
):
    if max_rs_steps is None:
        return _rs_local_impl_orig(sample_and_accept_fn, z_init, subsample_idx, rs_key, resample_key, params, max_rs_steps, rs_particles)
    else:
        return _rs_local_impl_max(sample_and_accept_fn, z_init, subsample_idx, rs_key, resample_key, params, max_rs_steps, rs_particles)


def _rs_local_impl_max(sample_and_accept_fn, z_init, subsample_idx, rs_key, resample_key, params, max_rs_steps, rs_particles):
    del z_init
    # sample_and_accept_fn(idx, key, param) -> z, (accept_lp, lw, guide_lp)
    S = rs_particles
    M = subsample_idx.shape[0]
    batch_key_q = jax.random.split(rs_key, max_rs_steps * M).reshape((max_rs_steps, M, 2))
    batch_z, (batch_accept_log_prob, batch_log_weight, batch_guide_lp) = jax.vmap(sample_and_accept_fn, in_axes=(None, 0, None))(
        subsample_idx, batch_key_q, params
    )
    assert batch_accept_log_prob.shape == (max_rs_steps, M)
    log_u = -random.exponential(resample_key, shape=(max_rs_steps, M))
    batch_is_accepted = log_u < batch_accept_log_prob
    maybe_accept_indices = jnp.argsort(batch_is_accepted, axis=0)[-S:]
    key_q, z, log_weight, guide_lp, is_accepted = jax.tree_util.tree_map(
        lambda x: _subsample_indexing(x, maybe_accept_indices),
        (batch_key_q, batch_z, batch_log_weight, batch_guide_lp, batch_is_accepted)
    )
    log_Z = logsumexp(batch_accept_log_prob, axis=0) - jnp.log(max_rs_steps)
    first_log_a = batch_accept_log_prob[:S]
    return key_q, z, log_weight, log_Z, first_log_a, guide_lp, is_accepted


def _rs_local_impl_orig(
    sample_and_accept_fn, z_init, subsample_idx, rs_key, resample_key, params, max_rs_steps, rs_particles
):
    del max_rs_steps
    # sample_and_accept_fn(idx, key, param) -> z, (accept_lp, lw, guide_lp)
    keys = random.split(rs_key, rs_particles)
    S = rs_particles
    M = subsample_idx.shape[0]
    assert z_init.ndim == 2
    assert z_init.shape[0] == M
    zs_init = tree_map(lambda x: jnp.broadcast_to(x, (S,) + jnp.shape(x)), z_init)
    neg_inf = jnp.full((S, M), -jnp.inf)
    keys = jax.vmap(lambda k: random.split(k, M))(keys)
    buffer = (keys, zs_init, neg_inf, neg_inf, jnp.full((S, M), False))
    init_val = (0, resample_key, keys, neg_inf, neg_inf, jnp.full(M, 0), buffer)

    def cond_fn(val):
        is_accepted = val[-1][-1]
        keep_running = is_accepted.sum(0) < S
        return keep_running.any()

    def body_fn(key, resample_idx):
        key_next, key_uniform, key_q = jax.vmap(
            lambda k: random.split(k, 3), out_axes=1
        )(key)
        z, (accept_log_prob, log_weight, guide_lp) = sample_and_accept_fn(
            subsample_idx[resample_idx], key_q, params
        )
        log_u = -jax.vmap(random.exponential)(key_uniform)
        is_accepted = log_u < accept_log_prob
        return key_next, accept_log_prob, (key_q, z, log_weight, guide_lp, is_accepted)

    def batch_body_fn(val):
        (
            step,
            resample_key,
            keys,
            batch_log_a_sum,
            batch_first_log_a,
            batch_num_samples,
            batch_buffer,
        ) = val
        if resample_key is not None:
            resample_key, resample_subkey = random.split(resample_key)
            # distribute batch-size resource to subsample items
            is_accepted = batch_buffer[-1]
            weights = S - is_accepted.sum(0)
            weights = jnp.where(weights < 0, 0, weights)
            weights = weights / weights.sum(-1, keepdims=True)
            # Use categorical is faster than systematic
            # resample_idxs = jax.random.categorical(resample_subkey, jnp.log(weights), shape=(M,))
            resample_idxs = get_systematic_resampling_indices(weights, resample_subkey, M)
        else:
            resample_idxs = jnp.arange(M)
        assert resample_idxs.shape == (M,)

        keys_next, batch_accept_log_prob, batch_candidate = jax.vmap(
            body_fn, in_axes=(0, None)
        )(keys, resample_idxs)

        def update_idx(i, val):
            batch_log_a_sum, batch_first_log_a, batch_num_samples, batch_buffer = val
            idx = resample_idxs[i]
            accept_log_prob = batch_accept_log_prob[:, i]
            candidate = tree_map(lambda x: x[:, i], batch_candidate)
            buffer = tree_map(lambda x: x[:, idx], batch_buffer)
            log_a_sum = batch_log_a_sum[:, idx]
            first_log_a = batch_first_log_a[:, idx]
            num_samples = batch_num_samples[idx]

            buffer_extend = tree_map(
                lambda a, b: jnp.concatenate([a, b]), candidate, buffer
            )
            is_accepted = buffer_extend[-1]
            maybe_accept_indices = jnp.argsort(is_accepted)[-S:]
            new_buffer = tree_map(lambda x: x[maybe_accept_indices], buffer_extend)
            log_a_sum = logsumexp(
                jnp.stack([log_a_sum, accept_log_prob], axis=0), axis=0
            )
            first_log_a = select(num_samples == 0, accept_log_prob, first_log_a)

            batch_buffer = tree_map(
                lambda x, y: x.at[:, idx].set(y), batch_buffer, new_buffer
            )
            batch_log_a_sum = batch_log_a_sum.at[:, idx].set(log_a_sum)
            batch_first_log_a = batch_first_log_a.at[:, idx].set(first_log_a)
            batch_num_samples = batch_num_samples.at[idx].set(num_samples + 1)
            return batch_log_a_sum, batch_first_log_a, batch_num_samples, batch_buffer

        if resample_key is not None:
            (
                batch_log_a_sum,
                batch_first_log_a,
                batch_num_samples,
                batch_buffer,
            ) = lax.fori_loop(
                0,
                M,
                update_idx,
                (batch_log_a_sum, batch_first_log_a, batch_num_samples, batch_buffer),
            )
        else:
            buffer_extend = tree_map(
                lambda a, b: jnp.concatenate([a, b]), batch_candidate, batch_buffer
            )
            is_accepted = buffer_extend[-1]
            maybe_accept_indices = jnp.argsort(is_accepted, axis=0)[-S:]
            batch_buffer = tree_map(lambda x: _subsample_indexing(x,maybe_accept_indices), buffer_extend)

            batch_log_a_sum = logsumexp(jnp.stack([batch_log_a_sum, batch_accept_log_prob], axis=0), axis=0)
            batch_first_log_a = jnp.where(batch_num_samples == 0, batch_accept_log_prob, batch_first_log_a)
            batch_num_samples = batch_num_samples + 1
        return (
            step + 1,
            resample_key,
            keys_next,
            batch_log_a_sum,
            batch_first_log_a,
            batch_num_samples,
            batch_buffer,
        )

    _, _, _, log_a_sum, first_log_a, num_samples, buffer = jax.lax.while_loop(
        cond_fn, batch_body_fn, init_val
    )
    key_q, z, log_w, guide_lp, is_accepted = buffer
    assert is_accepted.shape == (S, M)
    log_Z = logsumexp(log_a_sum, axis=0) - jnp.log(num_samples * S)
    return key_q, z, log_w, log_Z, first_log_a, guide_lp, is_accepted


@partial(jax.custom_vjp, nondiff_argnums=(0, 6, 7))
def _rs_local_custom_impl(
    sample_and_accept_fn, z_init, subsample_idx, rs_key, resample_key, params, max_rs_steps, rs_particles
):
    return _rs_local_impl(
        sample_and_accept_fn, z_init, subsample_idx, rs_key, resample_key, params, max_rs_steps, rs_particles
    )


def _rs_local_fwd(
    sample_and_accept_fn, z_init, subsample_idx, keys, resample_key, params, max_rs_steps, rs_particles
):
    out = _rs_local_custom_impl(
        sample_and_accept_fn, z_init, subsample_idx, keys, resample_key, params, max_rs_steps, rs_particles
    )
    key_q = out[0]
    return out, (subsample_idx, key_q, params)


def _rs_local_bwd(sample_and_accept_fn, max_rs_steps, rs_particles, res, g):
    subsample_idx, key_q, params = res
    _, z_grads, lw_grads, *_ = g

    def get_z_and_lw(key, params):
        z, (_, lw, _) = sample_and_accept_fn(subsample_idx, key, params)
        return z, lw

    def sample_grad(key, z_grad, lw_grad):
        _, guide_vjp = jax.vjp(partial(get_z_and_lw, key), params)
        return guide_vjp((z_grad, lw_grad))[0]

    batch_params_grad = jax.vmap(sample_grad)(key_q, z_grads, lw_grads)
    params_grad = jax.tree_util.tree_map(lambda x: x.sum(0), batch_params_grad)
    return (None, None, None, None, params_grad)


_rs_local_custom_impl.defvjp(_rs_local_fwd, _rs_local_bwd)
