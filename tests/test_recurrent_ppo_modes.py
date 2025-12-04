#!/usr/bin/env python3
"""
CI Smoke Test for 3-Level Recurrent PPO

Tests that all three recurrent modes (cached, truncated, full) run without errors:
- Level 1 (cached): No BPTT, fast but approximate
- Level 2 (truncated): K-step BPTT windows
- Level 3 (full): Full-sequence BPTT

This is NOT a performance benchmark - it's a correctness sanity check.
If this fails, recurrent PPO logic is broken.
"""

import torch
import pytest
from pseudo_mamba.envs.delayed_cue import DelayedCueEnv
from pseudo_mamba.controllers.gru import GRUController
from pseudo_mamba.rlh.actor_critic import ActorCritic
from pseudo_mamba.rlh.ppo import PPO
from pseudo_mamba.rlh.rollout import RolloutBuffer


@pytest.fixture
def device():
    """Use CPU for CI tests to avoid GPU requirements."""
    return torch.device("cpu")


@pytest.fixture
def env(device):
    """Small delayed cue environment for testing."""
    return DelayedCueEnv(batch_size=4, device=device, sequence_length=32)


@pytest.fixture
def actor_critic(env, device):
    """GRU-based actor-critic model."""
    controller = GRUController(
        input_dim=env.obs_dim,
        hidden_dim=64,
        feature_dim=64
    )
    return ActorCritic(controller, act_dim=env.act_dim).to(device)


def collect_rollout(env, actor_critic, horizon=32, device=torch.device("cpu")):
    """
    Collect a single rollout for testing.
    Returns a filled RolloutBuffer.
    """
    buffer = RolloutBuffer(
        num_steps=horizon,
        num_envs=env.batch_size,
        obs_dim=env.obs_dim,
        act_dim=env.act_dim,
        device=device
    )

    obs = env.reset()
    state = actor_critic.init_state(env.batch_size, device)
    buffer.states[0] = state

    for step in range(horizon):
        with torch.no_grad():
            logits, value, new_state = actor_critic.forward_step(obs, state)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            logprob = dist.log_prob(action)

        next_obs, reward, done, info = env.step(action)
        buffer.insert(obs, action, reward, done, value.squeeze(-1), logprob, new_state)

        obs = next_obs
        state = new_state

    # Compute GAE
    with torch.no_grad():
        _, last_value, _ = actor_critic.forward_step(obs, state)
        buffer.compute_gae(last_value.squeeze(-1))

    return buffer


def test_recurrent_mode_cached(env, actor_critic, device):
    """Test Level 1: Cached mode (no BPTT)."""
    ppo = PPO(
        actor_critic=actor_critic,
        lr=3e-4,
        recurrent_mode="cached",
        ppo_epochs=2,
        num_minibatches=2
    )

    buffer = collect_rollout(env, actor_critic, horizon=32, device=device)
    metrics = ppo.update(buffer)

    assert "loss" in metrics
    assert "pg_loss" in metrics
    assert "val_loss" in metrics
    assert metrics["loss"] >= 0, "Loss should be non-negative"
    print(f"[PASS] Cached mode: loss={metrics['loss']:.4f}")


def test_recurrent_mode_truncated(env, actor_critic, device):
    """Test Level 2: Truncated BPTT with K=8."""
    ppo = PPO(
        actor_critic=actor_critic,
        lr=3e-4,
        recurrent_mode="truncated",
        burn_in_steps=8,
        ppo_epochs=2,
        num_minibatches=2
    )

    buffer = collect_rollout(env, actor_critic, horizon=32, device=device)
    metrics = ppo.update(buffer)

    assert "loss" in metrics
    assert "pg_loss" in metrics
    assert "val_loss" in metrics
    assert metrics["loss"] >= 0, "Loss should be non-negative"
    print(f"[PASS] Truncated mode (K=8): loss={metrics['loss']:.4f}")


def test_recurrent_mode_full(env, actor_critic, device):
    """Test Level 3: Full-sequence BPTT."""
    ppo = PPO(
        actor_critic=actor_critic,
        lr=3e-4,
        recurrent_mode="full",
        ppo_epochs=2,
        num_minibatches=2
    )

    buffer = collect_rollout(env, actor_critic, horizon=32, device=device)
    metrics = ppo.update(buffer)

    assert "loss" in metrics
    assert "pg_loss" in metrics
    assert "val_loss" in metrics
    assert metrics["loss"] >= 0, "Loss should be non-negative"
    print(f"[PASS] Full BPTT mode: loss={metrics['loss']:.4f}")


def test_invalid_recurrent_mode(actor_critic):
    """Test that invalid recurrent mode raises ValueError."""
    with pytest.raises(ValueError, match="Invalid recurrent_mode"):
        PPO(
            actor_critic=actor_critic,
            recurrent_mode="invalid_mode"
        )


def test_all_modes_consistency(env, actor_critic, device):
    """
    Smoke test: All three modes should run on the same rollout.
    This doesn't check correctness, just that nothing crashes.
    """
    buffer = collect_rollout(env, actor_critic, horizon=32, device=device)

    modes = ["cached", "truncated", "full"]
    for mode in modes:
        ppo = PPO(
            actor_critic=actor_critic,
            lr=3e-4,
            recurrent_mode=mode,
            burn_in_steps=8,
            ppo_epochs=1,
            num_minibatches=2
        )
        metrics = ppo.update(buffer)
        assert "loss" in metrics, f"Mode {mode} failed to return metrics"
        print(f"[PASS] Mode '{mode}' completed successfully")


if __name__ == "__main__":
    # Allow running directly for quick testing
    print("Running recurrent PPO smoke tests...")

    device = torch.device("cpu")
    env = DelayedCueEnv(batch_size=4, device=device, sequence_length=32)

    controller = GRUController(input_dim=env.obs_dim, hidden_dim=64, feature_dim=64)
    actor_critic = ActorCritic(controller, act_dim=env.act_dim).to(device)

    print("\n1. Testing cached mode...")
    test_recurrent_mode_cached(env, actor_critic, device)

    print("\n2. Testing truncated mode...")
    test_recurrent_mode_truncated(env, actor_critic, device)

    print("\n3. Testing full BPTT mode...")
    test_recurrent_mode_full(env, actor_critic, device)

    print("\n4. Testing invalid mode handling...")
    test_invalid_recurrent_mode(actor_critic)

    print("\n5. Testing all modes consistency...")
    test_all_modes_consistency(env, actor_critic, device)

    print("\n✅ All recurrent PPO smoke tests passed!")
