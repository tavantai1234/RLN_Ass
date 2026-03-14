import argparse
import os
import numpy as np
import gymnasium as gym
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
from functools import partial
from flax.linen.initializers import constant, orthogonal
from gymnasium.wrappers import AtariPreprocessing
import ale_py  # needed for ALE env registration

ATARI_MAX_FRAMES = int(108000 / 4)


def make_eval_env(env_id: str, seed: int, record_video: bool, video_dir: str):
    env = gym.make(env_id, frameskip=1)
    env = AtariPreprocessing(env, noop_max=30, frame_skip=4, scale_obs=False)
    env = gym.wrappers.FrameStackObservation(env, 4)  # -> (4, 84, 84)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=ATARI_MAX_FRAMES)

    if record_video:
        os.makedirs(video_dir, exist_ok=True)
        # Record every episode
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=video_dir,
            episode_trigger=lambda ep: True,
            name_prefix="beamrider",
        )

    env.reset(seed=seed)
    return env


class Network(nn.Module):
    @nn.compact
    def __call__(self, x):
        # x: (B, 4, 84, 84) -> (B, 84, 84, 4)
        x = jnp.transpose(x, (0, 2, 3, 1))
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


@partial(jax.jit, static_argnums=(2,))
def act(params, obs, action_dim: int):
    obs = jnp.asarray(obs)
    h = Network().apply(params["network_params"], obs)
    logits = Actor(action_dim).apply(params["actor_params"], h)
    return jnp.argmax(logits, axis=1)


def _get_by_numeric_keys(d, n: int):
    """Try to read d['0'], d['1'], ..., d[str(n-1)]"""
    out = []
    for i in range(n):
        k = str(i)
        if k not in d:
            return None
        out.append(d[k])
    return out


def load_cleanrl_model(model_path: str):
    with open(model_path, "rb") as f:
        payload = flax.serialization.from_bytes(None, f.read())

    # payload can be list/tuple OR dict with numeric keys
    if isinstance(payload, dict):
        # Often payload is {"0": hparams, "1": weights_blob}
        if "0" in payload and "1" in payload:
            saved_hparams = payload["0"]
            blob = payload["1"]
        else:
            # fallback: take first two values
            items = list(payload.items())
            saved_hparams = items[0][1]
            blob = items[1][1]
    else:
        saved_hparams, blob = payload

    # Now blob might be:
    # A) list/tuple [net, actor, critic]
    # B) dict {"0": net, "1": actor, "2": critic}
    # C) FrozenDict/dict {"network_params":..., "actor_params":..., "critic_params":...}

    if isinstance(blob, (list, tuple)) and len(blob) == 3:
        network_params, actor_params, critic_params = blob

    elif isinstance(blob, dict):
        triple = _get_by_numeric_keys(blob, 3)
        if triple is not None:
            network_params, actor_params, critic_params = triple
        elif all(k in blob for k in ["network_params", "actor_params", "critic_params"]):
            network_params = blob["network_params"]
            actor_params = blob["actor_params"]
            critic_params = blob["critic_params"]
        else:
            raise ValueError(f"Unknown blob dict keys: {list(blob.keys())[:30]}")

    else:
        raise ValueError(f"Unknown blob type: {type(blob)}")

    params = flax.core.FrozenDict({
        "network_params": network_params,
        "actor_params": actor_params,
        "critic_params": critic_params,
    })
    return saved_hparams, params


def maybe_press_fire(env, action_dim: int):
    """
    Some Atari games need FIRE to start.
    BeamRider often uses FIRE=1 in minimal action set (not guaranteed).
    This helper tries action=1 for a few steps, safely.
    """
    if action_dim <= 1:
        return

    for _ in range(3):
        obs, r, terminated, truncated, _ = env.step(1)
        if terminated or truncated:
            break


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", type=str, required=True)
    p.add_argument("--env-id", type=str, default="ALE/BeamRider-v5")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--record-video", action="store_true")
    p.add_argument("--video-dir", type=str, default="videos_play")
    p.add_argument("--press-fire", action="store_true", help="try FIRE at episode start")
    args = p.parse_args()

    saved_hparams, params = load_cleanrl_model(args.model_path)
    print("Loaded model from:", args.model_path)
    try:
        print("Saved hparams keys:", list(saved_hparams.keys())[:20])
    except Exception:
        pass

    env = make_eval_env(args.env_id, args.seed, args.record_video, args.video_dir)
    action_dim = env.action_space.n
    print("Action dim:", action_dim)

    returns = []
    lengths = []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        if args.press_fire:
            maybe_press_fire(env, action_dim)
            obs, _ = env.reset(seed=args.seed + ep)  # reset again for clean start

        obs = np.asarray(obs)[None, ...]  # (1,4,84,84)
        done = False
        total = 0.0
        length = 0

        while not done:
            a = int(np.array(act(params, obs, action_dim))[0])
            obs, r, terminated, truncated, _ = env.step(a)
            obs = np.asarray(obs)[None, ...]
            total += float(r)
            length += 1
            done = bool(terminated or truncated)

        returns.append(total)
        lengths.append(length)
        print(f"episode {ep+1:02d}: return={total:.1f}, length={length}")

    print(f"\nMean return over {args.episodes} eps: {np.mean(returns):.2f} ± {np.std(returns):.2f}")
    print(f"Mean length: {np.mean(lengths):.1f}")

    env.close()

    if args.record_video:
        print(f"\nVideos saved to: {os.path.abspath(args.video_dir)}")


if __name__ == "__main__":
    main()