"""Native strategy CPU contracts; CUDA rasterization is explicitly opt-in."""
import math
import os
import sys
from types import SimpleNamespace

import pytest

from tools.streetview_engine.training import (TrainingSettings, create_strategy,
    default_reset_compatibility, strategy_pre_backward, strategy_post_backward)


def reset_opa(**kwargs):
    kwargs["state"]["test_native_resets"] = kwargs["state"].get("test_native_resets", 0) + 1


class BrokenReset:
    def step_post_backward(self, params, optimizers, state, step, info, packed=False):
        if step % self.reset_every == 0 & step > 0:
            reset_opa(state=state)


class FixedReset:
    def step_post_backward(self, params, optimizers, state, step, info, packed=False):
        if step % self.reset_every == 0 and step > 0:
            reset_opa(state=state)


class UnknownReset:
    def step_post_backward(self, params, optimizers, state, step, info, packed=False):
        if step > 3:
            reset_opa(state=state)


@pytest.mark.parametrize("value", [None, True, "default", "3DGS", "", 3, []])
def test_reject_unknown_strategy(value):
    with pytest.raises(ValueError, match="strategy"):
        TrainingSettings(strategy=value)


def test_strategy_is_opt_in_and_regularization_never_changes_implicitly():
    baseline = TrainingSettings()
    alternative = TrainingSettings.from_inputs({}, {"training": {"strategy": "3dgs"}})
    assert baseline.strategy == "mcmc"
    assert alternative.strategy == "3dgs"
    assert (baseline.opacity_reg, baseline.scale_reg) == (.01, .01)
    assert (alternative.opacity_reg, alternative.scale_reg) == (.01, .01)
    explicit = TrainingSettings(strategy="3dgs", opacity_reg=0., scale_reg=0.)
    assert explicit.opacity_reg == explicit.scale_reg == 0.


@pytest.mark.parametrize("steps", [30, 6000, 30000])
def test_mcmc_constructor_and_callback_sequence_unchanged(monkeypatch, steps):
    class NativeMCMC:
        def __init__(self, **kwargs):
            self.received = kwargs
        def initialize_state(self):
            return {"native": True}
        def check_sanity(self, params, optimizers):
            assert params == {"means": [0]} and optimizers == {}
        def step_post_backward(self, *args, **kwargs):
            self.post = args, kwargs
    monkeypatch.setitem(sys.modules, "gsplat", SimpleNamespace(MCMCStrategy=NativeMCMC))
    options = TrainingSettings(steps=steps)
    params, optimizers = {"means": [0]}, {}
    strategy, state = create_strategy(options, params, optimizers)
    factor = steps/30000
    assert strategy.received == dict(cap_max=1000000, noise_lr=500000.,
        refine_start_iter=max(1, round(500*factor)), refine_stop_iter=max(2, round(25000*factor)),
        refine_every=max(1, round(100*factor)), verbose=False)
    strategy_pre_backward(options, strategy, params, optimizers, state, 10, {})
    strategy_post_backward(options, strategy, params, optimizers, state, 10, {}, lr=.001)
    assert strategy.post == ((params, optimizers, state, 10, {}), {"lr": .001})


def test_known_reset_ast_and_unknown_source_are_explicit():
    broken = default_reset_compatibility(BrokenReset())
    fixed = default_reset_compatibility(FixedReset())
    assert broken["apply_missing_reset"] is True
    assert fixed["apply_missing_reset"] is False
    assert len(broken["native_post_backward_sha256"]) == 64
    assert broken["native_post_backward_sha256"] != fixed["native_post_backward_sha256"]
    with pytest.raises(RuntimeError, match="Unrecognized native"):
        default_reset_compatibility(UnknownReset())


@pytest.mark.parametrize("native_class", [BrokenReset, FixedReset])
def test_compatibility_resets_once_without_changing_pruning_cadence(monkeypatch, native_class):
    strategy = native_class()
    strategy.reset_every, strategy.refine_stop_iter = 3, 9
    strategy.refine_scale2d_stop_iter = 0
    strategy.grow_grad2d, strategy.grow_scale2d, strategy.prune_opa = .0002, .05, .005
    state = {"_streetview_reset_compatibility": default_reset_compatibility(strategy)}
    monkeypatch.setitem(sys.modules, "gsplat.strategy.ops", SimpleNamespace(reset_opa=reset_opa))
    options = TrainingSettings(strategy="3dgs", max_splats=4)
    # Native callbacks normally return at refine_stop; the fixed synthetic
    # callback models only its reset clause, so test the shared active horizon.
    for step in range(9):
        strategy_post_backward(options, strategy, {"means": [0]*3}, {}, state, step, {}, lr=0.)
        assert strategy.reset_every == 3
        assert (strategy.grow_grad2d, strategy.grow_scale2d) == (.0002, .05)
    assert state["test_native_resets"] == 2
    assert state["_streetview_reset_compatibility"]["resets_applied"] == (2 if native_class is BrokenReset else 0)


def test_budget_guard_restores_thresholds_on_error_and_fails_without_truncation():
    class Strategy:
        refine_scale2d_stop_iter = 0
        grow_grad2d, grow_scale2d = .0002, .05
        def step_post_backward(self, params, *args, **kwargs):
            assert kwargs == {"packed": True}
            assert math.isinf(self.grow_grad2d) and math.isinf(self.grow_scale2d)
            raise RuntimeError("native error")
    options = TrainingSettings(strategy="3dgs", max_splats=6)
    strategy, params = Strategy(), {"means": list(range(4))}
    with pytest.raises(RuntimeError, match="native error"):
        strategy_post_backward(options, strategy, params, {}, {}, 1, {}, lr=0.)
    assert (strategy.grow_grad2d, strategy.grow_scale2d) == (.0002, .05)
    assert params["means"] == list(range(4))
    with pytest.raises(RuntimeError, match="before native"):
        strategy_post_backward(options, strategy, {"means": list(range(7))}, {}, {}, 1, {}, lr=0.)
    strategy.refine_scale2d_stop_iter = 1
    with pytest.raises(ValueError, match="2D-radius"):
        strategy_post_backward(options, strategy, params, {}, {}, 1, {}, lr=0.)


def native_fixture(count=4, device="cpu"):
    torch = pytest.importorskip("torch")
    pytest.importorskip("gsplat")
    means = torch.zeros((count, 3), device=device)
    means[:, 0] = torch.linspace(-.2, .2, count, device=device)
    means[:, 2] = 2.
    sizes = torch.tensor([.005 if i % 2 == 0 else .02 for i in range(count)], device=device)
    params = torch.nn.ParameterDict({
        "means": torch.nn.Parameter(means),
        "scales": torch.nn.Parameter(sizes.log()[:, None].repeat(1, 3)),
        "quats": torch.nn.Parameter(torch.tensor([[1., 0, 0, 0]], device=device).repeat(count, 1)),
        "opacities": torch.nn.Parameter(torch.zeros(count, device=device)),
        "sh0": torch.nn.Parameter(torch.zeros((count, 1, 3), device=device)),
        "shN": torch.nn.Parameter(torch.zeros((count, 8, 3), device=device)),
    })
    optimizers = {key: torch.optim.Adam([value], lr=0.) for key, value in params.items()}
    return torch, params, optimizers


def native_step(options, strategy, params, optimizers, state, step):
    torch = pytest.importorskip("torch")
    means2d = params["means"][:, :2]*2.
    count = len(params["means"])
    info = dict(width=32, height=32, n_cameras=1, gaussian_ids=torch.arange(count),
                radii=torch.ones((count, 2)), means2d=means2d)
    strategy_pre_backward(options, strategy, params, optimizers, state, step, info)
    means2d.sum().backward()
    assert torch.isfinite(means2d.grad).all() and means2d.grad.abs().sum() > 0
    for optimizer in optimizers.values():
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    strategy_post_backward(options, strategy, params, optimizers, state, step, info, lr=0.)


@pytest.mark.parametrize("steps", [30, 6000, 30000])
def test_native_schedule_and_normalized_scene_scale(steps):
    _, params, optimizers = native_fixture()
    strategy, state = create_strategy(TrainingSettings(strategy="3dgs", steps=steps), params, optimizers)
    factor = steps/30000
    assert strategy.refine_start_iter == max(1, round(500*factor))
    assert strategy.refine_stop_iter == max(2, round(15000*factor))
    assert strategy.refine_every == max(1, round(100*factor))
    assert strategy.reset_every == max(1, round(3000*factor))
    assert strategy.refine_scale2d_stop_iter == 0 and not strategy.absgrad and not strategy.revised_opacity
    assert state["scene_scale"] == 1.


@pytest.mark.parametrize("cap, expected", [(8, 8), (7, 4)])
def test_actual_native_duplicate_split_event_obeys_even_and_odd_cap(cap, expected):
    torch, params, optimizers = native_fixture()
    options = TrainingSettings(strategy="3dgs", max_splats=cap, steps=30000)
    strategy, state = create_strategy(options, params, optimizers)
    # At 600 the official event duplicates two small and splits two large rows.
    native_step(options, strategy, params, optimizers, state, 600)
    assert len(params["means"]) == expected
    assert strategy.grow_grad2d == .0002
    assert state["grad2d"].shape == (expected,)
    assert torch.isfinite(params["scales"]).all()
    for key, optimizer in optimizers.items():
        assert optimizer.param_groups[0]["params"][0] is params[key]


def test_actual_native_guard_keeps_pruning_and_reset_enabled():
    torch, params, optimizers = native_fixture()
    options = TrainingSettings(strategy="3dgs", max_splats=6, steps=30000)
    strategy, state = create_strategy(options, params, optimizers)
    with torch.no_grad():
        params["opacities"][0] = -20.
    native_step(options, strategy, params, optimizers, state, 600)
    assert len(params["means"]) == 3  # Growth blocked, actual native opacity prune retained.
    native_step(options, strategy, params, optimizers, state, 3000)
    assert float(params["opacities"].sigmoid().max()) <= .010001
    # Native physical-size pruning only starts AFTER reset_every, unchanged.
    with torch.no_grad():
        params["scales"][0] = math.log(.2)
    before = len(params["means"])
    native_step(options, strategy, params, optimizers, state, 3100)
    assert len(params["means"]) < before


@pytest.mark.skipif(os.environ.get("STREETVIEW_RUN_DEFAULT_STRATEGY_CUDA") != "1",
                    reason="Explicit opt-in actual packed CUDA DefaultStrategy integration")
def test_actual_cuda_packed_default_callbacks():
    torch, params, optimizers = native_fixture(device="cuda")
    from gsplat import rasterization
    options = TrainingSettings(strategy="3dgs", max_splats=8, steps=30000)
    strategy, state = create_strategy(options, params, optimizers)
    # Force a native growth event on an actual packed render gradient. No scene
    # data, quality claim, optimizer tuning or fake rasterizer is involved.
    strategy.grow_grad2d = 0.
    for step in (600, 700, 3000):
        for optimizer in optimizers.values():
            optimizer.zero_grad(set_to_none=True)
        rgb, alpha, info = rasterization(means=params["means"], quats=params["quats"],
            scales=params["scales"].exp(), opacities=params["opacities"].sigmoid(),
            colors=torch.cat((params["sh0"], params["shN"]), dim=1),
            viewmats=torch.eye(4, device="cuda")[None],
            Ks=torch.tensor([[[32., 0, 16], [0, 32., 16], [0, 0, 1]]], device="cuda"),
            width=32, height=32, sh_degree=0, packed=True)
        strategy_pre_backward(options, strategy, params, optimizers, state, step, info)
        ramp = torch.linspace(.2, .8, 32, device="cuda")[None, None, :, None]
        loss = ((rgb-ramp)**2).mean() + .1*alpha.mean()
        loss.backward()
        assert info["means2d"].grad is not None and info["means2d"].grad.norm() > 0
        assert all(torch.isfinite(value.grad).all() for value in params.values() if value.grad is not None)
        for optimizer in optimizers.values():
            optimizer.step()
        strategy_post_backward(options, strategy, params, optimizers, state, step, info, lr=0.)
        assert 4 <= len(params["means"]) <= options.max_splats
    assert len(params["means"]) == 8
    assert float(params["opacities"].sigmoid().max()) <= .010001
    torch.cuda.synchronize()
