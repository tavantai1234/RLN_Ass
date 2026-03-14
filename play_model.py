import argparse
import numpy as np
import gymnasium as gym
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.linen.initializers import constant, orthogonal
from gymnasium.wrappers import AtariPreprocessing
import ale_py  # needed for ALE env registration

ATARI_MAX_FRAMES = int(108000 / 4)

def make_eval_env(env_id, seed):
    env = gym.make(env_id, frameskip=1)
    env = AtariPreprocessing(env, noop_max=30, frame_skip=4, scale_obs=False)
    env = gym.wrappers.FrameStackObservation(env, 4)  # gives (4,84,84) usually
    env = gym.wrappers.TimeLimit(env, max_episode_steps=ATARI_MAX_FRAMES)
    env.reset(seed=seed)
    return env

class Network(nn.Module):
    @nn.compact
    def __call__(self, x):
        # expects (B, 4, 84, 84)
        x = jnp.transpose(x, (0, 2, 3, 1))  # -> (B,84,84,4)
        x = x / 255.0
        x = nn.Conv(32, (8, 8), (4, 4), padding="VALID",
                    kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        x = nn.Conv(64, (4, 4), (2, 2), padding="VALID",
                    kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        x = nn.Conv(64, (3, 3), (1, 1), padding="VALID",
                    kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(512, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        return x

class Actor(nn.Module):
    action_dim: int
    @nn.compact
    def __call__(self, x):
        return nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(x)

from functools import partial

@partial(jax.jit, static_argnums=(2,))
def act(params, obs, action_dim: int):
    obs = jnp.asarray(obs)
    h = Network().apply(params["network_params"], obs)
    logits = Actor(action_dim).apply(params["actor_params"], h)
    return jnp.argmax(logits, axis=1)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", type=str, required=True)
    p.add_argument("--env-id", type=str, default="ALE/BeamRider-v5")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--seed", type=int, default=123)
    args = p.parse_args()

    with open(args.model_path, "rb") as f:
        payload = flax.serialization.from_bytes(None, f.read())
    # payload structure you saved: [vars(args), [network_params, actor_params, critic_params]]
    saved_hparams, (network_params, actor_params, critic_params) = payload
    saved_hparams, blob = payload

# Case 1: blob is (network_params, actor_params, critic_params)
    if isinstance(blob, (list, tuple)) and len(blob) == 3:
        network_params, actor_params, critic_params = blob

    # Case 2: blob is [FrozenDict({...})]  (a list with 1 element)
    elif isinstance(blob, (list, tuple)) and len(blob) == 1:
        blob = blob[0]
        network_params = blob["network_params"]
        actor_params   = blob["actor_params"]
        critic_params  = blob["critic_params"]

    # Case 3: blob is FrozenDict({...})
    else:
        network_params = blob["network_params"]
        actor_params   = blob["actor_params"]
        critic_params  = blob["critic_params"]

    params = flax.core.FrozenDict({
        "network_params": network_params,
        "actor_params": actor_params,
        "critic_params": critic_params,
    })

    env = make_eval_env(args.env_id, args.seed)
    action_dim = env.action_space.n

    returns = []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        obs = np.asarray(obs)              # (4,84,84)
        obs = obs[None, ...]               # add batch -> (1,4,84,84)
        done = False
        total = 0.0
        while not done:
            a = np.array(act(params, obs, action_dim))[0]
            obs, r, terminated, truncated, _ = env.step(a)
            obs = np.asarray(obs)[None, ...]
            total += float(r)
            done = bool(terminated or truncated)
        returns.append(total)
        print(f"episode {ep+1}: return={total:.1f}")

    print(f"\nMean return over {args.episodes} eps: {np.mean(returns):.2f} ± {np.std(returns):.2f}")
    env.close()

if __name__ == "__main__":
    main()