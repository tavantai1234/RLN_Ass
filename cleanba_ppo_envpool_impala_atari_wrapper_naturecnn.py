import argparse
import os
import random
import time
import uuid
from collections import deque
from distutils.util import strtobool
from functools import partial


os.environ[
    "XLA_PYTHON_CLIENT_MEM_FRACTION"
] = "0.6"  # see https://github.com/google/jax/discussions/6332#discussioncomment-1279991
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false " "intra_op_parallelism_threads=1"
import queue
import threading

import flax
import flax.linen as nn
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from tensorboardX import SummaryWriter
from gymnasium.vector import SyncVectorEnv
from gymnasium.wrappers import AtariPreprocessing
import ale_py

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="weather to capture videos of the agent performances (check out `videos` folder)")
    parser.add_argument("--save-model", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to save model into the `runs/{run_name}` folder")
    parser.add_argument("--upload-model", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to upload the saved model to huggingface")
    parser.add_argument("--hf-entity", type=str, default="",
        help="the user or org name of the model repository from the Hugging Face Hub")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="ALE/BeamRider-v5",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=50000000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--local-num-envs", type=int, default=60,
        help="the number of parallel game environments")
    parser.add_argument("--async-batch-size", type=int, default=20,
        help="legacy EnvPool argument — unused by SyncVectorEnv, kept for CLI compatibility")
    parser.add_argument("--clip-rewards", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, clip training rewards to [-1, 1]; evaluation always uses raw rewards")
    parser.add_argument("--num-steps", type=int, default=128,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=4,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=4,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.1,
        help="the surrogate clipping coefficient")
    parser.add_argument("--ent-coef", type=float, default=0.01,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")

    parser.add_argument("--actor-device-ids", type=int, nargs="+", default=[0], # type is actually List[int]
        help="the device ids that actor workers will use (currently only support 1 device)")
    parser.add_argument("--learner-device-ids", type=int, nargs="+", default=[0], # type is actually List[int]
        help="the device ids that learner workers will use")
    parser.add_argument("--distributed", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to use `jax.distirbuted`")
    parser.add_argument("--profile", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to call block_until_ready() for profiling")
    parser.add_argument("--test-actor-learner-throughput", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to test actor-learner throughput by removing the actor-learner communication")
    args = parser.parse_args()
    args.local_batch_size = int(args.local_num_envs * args.num_steps)
    args.local_minibatch_size = int(args.local_batch_size // args.num_minibatches)
    args.num_updates = args.total_timesteps // args.local_batch_size
    assert len(args.actor_device_ids) == 1, "only 1 actor_device_ids is supported now"
    # fmt: on
    return args


ATARI_MAX_FRAMES = int(
    108000 / 4
)  # 108000 is the max number of frames in an Atari game, divided by 4 to account for frame skipping


def make_env(env_id, seed, num_envs, async_batch_size=None):
    """
    Create a vectorized environment using Gymnasium SyncVectorEnv.
    Produces observations in NHWC format (B, 84, 84, 4).
    async_batch_size is ignored (kept for API compatibility).
    """
    def make_single_env(seed_offset):
        def thunk():
            # frameskip=1 required so AtariPreprocessing controls frame skipping (avoids 4x4=16x skip)
            env = gym.make(env_id, frameskip=1)
            env = AtariPreprocessing(env, noop_max=30, frame_skip=4, scale_obs=False)
            # Stack 4 frames: obs shape becomes (84, 84, 4) in NHWC format
            # Flax Conv expects NHWC by default
            env = gym.wrappers.FrameStackObservation(env, 4)
            env = gym.wrappers.TimeLimit(env, max_episode_steps=ATARI_MAX_FRAMES)
            env.reset(seed=seed + seed_offset)
            return env
        return thunk
    
    env_fns = [make_single_env(i) for i in range(num_envs)]
    return lambda: SyncVectorEnv(env_fns)


class Network(nn.Module):
    @nn.compact
    def __call__(self, x):
        # Input x is (B, 84, 84, 4) in NHWC format from FrameStackObservation
        # Flax Conv uses channels-last (NHWC) format by default
        x = x / (255.0)
        x = nn.Conv(
            32,
            kernel_size=(8, 8),
            strides=(4, 4),
            padding="VALID",
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = nn.relu(x)
        x = nn.Conv(
            64,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding="VALID",
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = nn.relu(x)
        x = nn.Conv(
            64,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="VALID",
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(512, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        return x


class Critic(nn.Module):
    @nn.compact
    def __call__(self, x):
        return nn.Dense(1, kernel_init=orthogonal(1), bias_init=constant(0.0))(x)


class Actor(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x):
        return nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(x)


@partial(jax.jit, static_argnums=(3))
def get_action_and_value(
    params: flax.core.FrozenDict,
    next_obs: np.ndarray,
    key: jax.random.PRNGKey,
    action_dim: int,
):
    next_obs = jnp.array(next_obs)
    hidden = Network().apply(params["network_params"], next_obs)
    logits = Actor(action_dim).apply(params["actor_params"], hidden)
    # sample action: Gumbel-softmax trick
    # see https://stats.stackexchange.com/questions/359442/sampling-from-a-categorical-distribution
    key, subkey = jax.random.split(key)
    u = jax.random.uniform(subkey, shape=logits.shape)
    action = jnp.argmax(logits - jnp.log(-jnp.log(u)), axis=1)
    logprob = jax.nn.log_softmax(logits)[jnp.arange(action.shape[0]), action]
    value = Critic().apply(params["critic_params"], hidden)
    return next_obs, action, logprob, value.squeeze(), key


def prepare_data(
    obs: list,
    dones: list,
    values: list,
    actions: list,
    logprobs: list,
    rewards: list,
    gamma: float = 0.99,
    lambda_: float = 0.95,
):
    # With SyncVectorEnv all envs step in lockstep, so data is already aligned.
    # shapes after stacking: (num_steps+1, num_envs, 84, 84, 4) — NHWC format, no env_id reordering needed.
    obs = jnp.asarray(obs)         # (T+1, B, 84, 84, 4)
    dones = jnp.asarray(dones)     # (T+1, B)
    values = jnp.asarray(values)   # (T+1, B)
    actions = jnp.asarray(actions) # (T+1, B)
    logprobs = jnp.asarray(logprobs) # (T+1, B)
    rewards = jnp.asarray(rewards) # (T+1, B)

    advantages, returns = compute_gae(rewards, values, dones, gamma, lambda_)  # (T, B) each

    # Extract first T steps (exclude the bootstrap value at T+1)
    T = obs.shape[0] - 1  # Explicit extraction instead of relying on global args
    b_obs = obs[:T].reshape((-1,) + obs.shape[2:])   # (T*B, 84, 84, 4)
    b_actions = actions[:T].reshape(-1)               # (T*B,)
    b_logprobs = logprobs[:T].reshape(-1)             # (T*B,)
    b_advantages = advantages.reshape(-1)             # (T*B,)
    b_returns = returns.reshape(-1)                   # (T*B,)
    return b_obs, b_actions, b_logprobs, b_advantages, b_returns


def rollout(
    key: jax.random.PRNGKey,
    args,
    rollout_queue,
    params_queue: queue.Queue,
    writer,
    learner_devices,
):
    envs = make_env(args.env_id, args.seed, args.local_num_envs, args.async_batch_size)()
    len_actor_device_ids = len(args.actor_device_ids)
    global_step = 0
    # TRY NOT TO MODIFY: start the game
    start_time = time.time()

    # put data in the last index
    episode_returns = np.zeros((args.local_num_envs,), dtype=np.float32)
    returned_episode_returns = np.zeros((args.local_num_envs,), dtype=np.float32)
    episode_lengths = np.zeros((args.local_num_envs,), dtype=np.float32)
    returned_episode_lengths = np.zeros((args.local_num_envs,), dtype=np.float32)
    next_obs, _ = envs.reset()
    next_obs = np.asarray(next_obs)  # materialise LazyFrames → contiguous np.ndarray
    # Transpose from NCHW (B, C, H, W) to NHWC (B, H, W, C) if needed
    if next_obs.shape[1] == 4 and next_obs.ndim == 4:
        next_obs = np.transpose(next_obs, (0, 2, 3, 1))  # (B, C, H, W) -> (B, H, W, C)
    # Verify observation shape is (B, 84, 84, 4); NHWC format
    assert next_obs.shape[-1] == 4, (
        f"Expected 4 stacked frames at last axis (NHWC), got {next_obs.shape}. "
        "After transpose, should be (B, 84, 84, 4) format."
    )

    params_queue_get_time = deque(maxlen=10)
    rollout_time = deque(maxlen=10)
    rollout_queue_put_time = deque(maxlen=10)
    actor_policy_version = 0
    for update in range(1, args.num_updates + 2):
        # NOTE: This is a major difference from the sync version:
        # at the end of the rollout phase, the sync version will have the next observation
        # ready for the value bootstrap, but the async version will not have it.
        # for this reason we do `num_steps + 1`` to get the extra states for value bootstrapping.
        # but note that the extra states are not used for the loss computation in the next iteration,
        # while the sync version will use the extra state for the loss computation.
        update_time_start = time.time()
        obs = []
        dones = []
        actions = []
        logprobs = []
        values = []
        rewards = []
        truncations = []
        terminations = []
        env_recv_time = 0
        inference_time = 0
        storage_time = 0
        env_send_time = 0

        # NOTE: `update != 2` is actually IMPORTANT — it allows us to start running policy collection
        # concurrently with the learning process. It also ensures the actor's policy version is only 1 step
        # behind the learner's policy version
        params_queue_get_time_start = time.time()
        if update != 2:
            params = params_queue.get()
            actor_policy_version += 1
        params_queue_get_time.append(time.time() - params_queue_get_time_start)
        writer.add_scalar("stats/params_queue_get_time", np.mean(params_queue_get_time), global_step)
        rollout_time_start = time.time()
        for step in range(args.num_steps + 1):  # num_steps + 1 to get the states for value bootstrapping.
            # Fix #1/#3: materialise LazyFrames and capture obs BEFORE stepping.
            # The policy sees obs_t; we store obs_t aligned with action_t/value_t/logprob_t.
            obs_t = np.asarray(next_obs)
            env_recv_time_start = time.time()
            inference_time_start = time.time()
            _, action, logprob, value, key = get_action_and_value(params, obs_t, key, envs.single_action_space.n)
            inference_time += time.time() - inference_time_start

            env_send_time_start = time.time()
            next_obs, reward, terminated, truncated, info = envs.step(np.array(action))
            # Transpose from NCHW (B, C, H, W) to NHWC (B, H, W, C) if needed
            next_obs = np.asarray(next_obs)
            if next_obs.shape[1] == 4 and next_obs.ndim == 4:
                next_obs = np.transpose(next_obs, (0, 2, 3, 1))  # (B, C, H, W) -> (B, H, W, C)
            env_send_time += time.time() - env_send_time_start

            env_recv_time += time.time() - env_recv_time_start
            global_step += args.local_num_envs * len_actor_device_ids * args.world_size

            storage_time_start = time.time()
            obs.append(obs_t)  # store pre-step obs, aligned with action/value/logprob
            dones.append(terminated | truncated)
            values.append(value)
            actions.append(action)
            logprobs.append(logprob)
            # Optionally clip rewards for training stability; raw reward used below for episode tracking
            train_reward = np.clip(reward, -1.0, 1.0) if args.clip_rewards else reward
            rewards.append(train_reward)
            truncations.append(truncated)
            terminations.append(terminated)

            # Vectorized episode tracking always uses raw (unclipped) rewards for reporting
            episode_returns += reward
            episode_lengths += 1
            done_mask = terminated | truncated
            returned_episode_returns = np.where(done_mask, episode_returns, returned_episode_returns)
            returned_episode_lengths = np.where(done_mask, episode_lengths, returned_episode_lengths)
            episode_returns *= (1.0 - done_mask.astype(np.float32))
            episode_lengths *= (1.0 - done_mask.astype(np.float32))

            storage_time += time.time() - storage_time_start
        
        if args.profile:
            action.block_until_ready()
        rollout_time.append(time.time() - rollout_time_start)
        writer.add_scalar("stats/rollout_time", np.mean(rollout_time), global_step)

        avg_episodic_return = np.mean(returned_episode_returns)
        writer.add_scalar("charts/avg_episodic_return", avg_episodic_return, global_step)
        writer.add_scalar("charts/avg_episodic_length", np.mean(returned_episode_lengths), global_step)
        print(f"global_step={global_step}, avg_episodic_return={avg_episodic_return}")
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        writer.add_scalar("stats/truncations", np.sum(truncations), global_step)
        writer.add_scalar("stats/terminations", np.sum(terminations), global_step)
        writer.add_scalar("stats/env_recv_time", env_recv_time, global_step)
        writer.add_scalar("stats/inference_time", inference_time, global_step)
        writer.add_scalar("stats/storage_time", storage_time, global_step)
        writer.add_scalar("stats/env_send_time", env_send_time, global_step)

        payload = (
            global_step,
            actor_policy_version,
            update,
            obs,
            dones,
            values,
            actions,
            logprobs,
            rewards,
        )
        if update == 1 or not args.test_actor_learner_throughput:
            rollout_queue_put_time_start = time.time()
            rollout_queue.put(payload)
            rollout_queue_put_time.append(time.time() - rollout_queue_put_time_start)
            writer.add_scalar("stats/rollout_queue_put_time", np.mean(rollout_queue_put_time), global_step)

        writer.add_scalar(
            "charts/SPS_update",
            int(
                args.local_num_envs
                * args.num_steps
                * len_actor_device_ids
                * args.world_size
                / (time.time() - update_time_start)
            ),
            global_step,
        )


@partial(jax.jit, static_argnums=(2))
def get_action_deterministic(
    params: flax.core.FrozenDict,
    obs: np.ndarray,
    action_dim: int,
):
    """Deterministic greedy policy for evaluation — picks argmax(logits)."""
    obs = jnp.array(obs)
    hidden = Network().apply(params["network_params"], obs)
    logits = Actor(action_dim).apply(params["actor_params"], hidden)
    return jnp.argmax(logits, axis=1)


@partial(jax.jit, static_argnums=(3))
def get_action_and_value2(
    params: flax.core.FrozenDict,
    x: np.ndarray,
    action: np.ndarray,
    action_dim: int,
):
    hidden = Network().apply(params["network_params"], x)
    logits = Actor(action_dim).apply(params["actor_params"], hidden)
    logprob = jax.nn.log_softmax(logits)[jnp.arange(action.shape[0]), action]
    logits = logits - jax.scipy.special.logsumexp(logits, axis=-1, keepdims=True)
    logits = logits.clip(min=jnp.finfo(logits.dtype).min)
    p_log_p = logits * jax.nn.softmax(logits)
    entropy = -p_log_p.sum(-1)
    value = Critic().apply(params["critic_params"], hidden).squeeze()
    return logprob, entropy, value


@jax.jit
def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    gamma: float = 0.99,
    lambda_: float = 0.95,
):
    # All inputs shape: (T+1, B) where T = num_steps
    # Compute advantages for steps 0..T-1; step T provides bootstrap value only.
    rewards = jnp.asarray(rewards)
    values = jnp.asarray(values)
    dones = jnp.asarray(dones)

    # Extract T from the input shape instead of relying on global args
    T = rewards.shape[0] - 1

    def gae_step(lastgaelam, x):
        done, value, next_value, reward = x
        nextnonterminal = 1.0 - done
        delta = reward + gamma * next_value * nextnonterminal - value
        lastgaelam = delta + gamma * lambda_ * nextnonterminal * lastgaelam
        return lastgaelam, lastgaelam

    # Scan in reverse from t=T-1 down to t=0
    xs = (
        jnp.flip(dones[:T], axis=0),       # done[t]
        jnp.flip(values[:T], axis=0),      # V(obs[t])
        jnp.flip(values[1:T + 1], axis=0), # V(obs[t+1]) — bootstrap
        jnp.flip(rewards[:T], axis=0),     # reward[t]
    )
    _, advantages_reversed = jax.lax.scan(
        gae_step, jnp.zeros(rewards.shape[1]), xs
    )
    advantages = jnp.flip(advantages_reversed, axis=0)  # (T, B)
    returns = advantages + values[:T]
    return advantages, returns


def ppo_loss(params, x, a, logp, mb_advantages, mb_returns, action_dim):
    newlogprob, entropy, newvalue = get_action_and_value2(params, x, a, action_dim)
    logratio = newlogprob - logp
    ratio = jnp.exp(logratio)
    approx_kl = ((ratio - 1) - logratio).mean()

    if args.norm_adv:
        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

    # Policy loss
    pg_loss1 = -mb_advantages * ratio
    pg_loss2 = -mb_advantages * jnp.clip(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
    pg_loss = jnp.maximum(pg_loss1, pg_loss2).mean()

    # Value loss
    v_loss = 0.5 * ((newvalue - mb_returns) ** 2).mean()

    entropy_loss = entropy.mean()
    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
    return loss, (pg_loss, v_loss, entropy_loss, jax.lax.stop_gradient(approx_kl))


@partial(jax.jit, static_argnums=(6))
def single_device_update(
    agent_state: TrainState,
    b_obs,
    b_actions,
    b_logprobs,
    b_advantages,
    b_returns,
    action_dim,
    key: jax.random.PRNGKey,
):
    ppo_loss_grad_fn = jax.value_and_grad(ppo_loss, has_aux=True)

    def update_epoch(carry, _):
        agent_state, key = carry
        key, subkey = jax.random.split(key)

        # taken from: https://github.com/google/brax/blob/main/brax/training/agents/ppo/train.py
        def convert_data(x: jnp.ndarray):
            x = jax.random.permutation(subkey, x)
            x = jnp.reshape(x, (args.num_minibatches, -1) + x.shape[1:])
            return x

        def update_minibatch(agent_state, minibatch):
            mb_obs, mb_actions, mb_logprobs, mb_advantages, mb_returns = minibatch
            (loss, (pg_loss, v_loss, entropy_loss, approx_kl)), grads = ppo_loss_grad_fn(
                agent_state.params,
                mb_obs,
                mb_actions,
                mb_logprobs,
                mb_advantages,
                mb_returns,
                action_dim,
            )
            agent_state = agent_state.apply_gradients(grads=grads)
            return agent_state, (loss, pg_loss, v_loss, entropy_loss, approx_kl, grads)

        agent_state, (loss, pg_loss, v_loss, entropy_loss, approx_kl, grads) = jax.lax.scan(
            update_minibatch,
            agent_state,
            (
                convert_data(b_obs),
                convert_data(b_actions),
                convert_data(b_logprobs),
                convert_data(b_advantages),
                convert_data(b_returns),
            ),
        )
        return (agent_state, key), (loss, pg_loss, v_loss, entropy_loss, approx_kl, grads)

    (agent_state, key), (loss, pg_loss, v_loss, entropy_loss, approx_kl, _) = jax.lax.scan(
        update_epoch, (agent_state, key), (), length=args.update_epochs
    )
    return agent_state, loss, pg_loss, v_loss, entropy_loss, approx_kl, key


if __name__ == "__main__":
    args = parse_args()
    if args.distributed:
        jax.distributed.initialize(
            local_device_ids=range(len(args.learner_device_ids) + len(args.actor_device_ids)),
        )
        print(list(range(len(args.learner_device_ids) + len(args.actor_device_ids))))

    args.world_size = jax.process_count()
    args.local_rank = jax.process_index()
    args.num_envs = args.local_num_envs * args.world_size
    args.batch_size = args.local_batch_size * args.world_size
    args.minibatch_size = args.local_minibatch_size * args.world_size
    args.num_updates = args.total_timesteps // (args.local_batch_size * args.world_size)
    # async_update not needed for SyncVectorEnv - removed
    local_devices = jax.local_devices()
    global_devices = jax.devices()
    learner_devices = [local_devices[d_id] for d_id in args.learner_device_ids]
    actor_devices = [local_devices[d_id] for d_id in args.actor_device_ids]
    global_learner_decices = [
        global_devices[d_id + process_index * len(local_devices)]
        for process_index in range(args.world_size)
        for d_id in args.learner_device_ids
    ]
    print("global_learner_decices", global_learner_decices)
    args.global_learner_decices = [str(item) for item in global_learner_decices]
    args.actor_devices = [str(item) for item in actor_devices]
    args.learner_devices = [str(item) for item in learner_devices]

    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{uuid.uuid4()}"
    if args.track and args.local_rank == 0:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)
    key, network_key, actor_key, critic_key = jax.random.split(key, 4)

    # env setup
    envs = make_env(args.env_id, args.seed, args.local_num_envs, args.async_batch_size)()
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    def linear_schedule(count):
        # anneal learning rate linearly after one training iteration which contains
        # (args.num_minibatches * args.update_epochs) gradient updates
        frac = 1.0 - (count // (args.num_minibatches * args.update_epochs)) / args.num_updates
        return args.learning_rate * frac

    network = Network()
    actor = Actor(action_dim=envs.single_action_space.n)
    critic = Critic()
    network_params = network.init(network_key, np.array([envs.single_observation_space.sample()]))
    agent_state = TrainState.create(
        apply_fn=None,
        params=flax.core.FrozenDict({
            'network_params': network_params,
            'actor_params': actor.init(actor_key, network.apply(network_params, np.array([envs.single_observation_space.sample()]))),
            'critic_params': critic.init(critic_key, network.apply(network_params, np.array([envs.single_observation_space.sample()]))),
        }),
        tx=optax.chain(
            optax.clip_by_global_norm(args.max_grad_norm),
            optax.inject_hyperparams(optax.adam)(
                learning_rate=linear_schedule if args.anneal_lr else args.learning_rate, eps=1e-5
            ),
        ),
    )
    agent_state = flax.jax_utils.replicate(agent_state, devices=learner_devices)

    # For single device, don't use pmap; call directly
    if len(learner_devices) == 1:
        def multi_device_update(agent_state, b_obs, b_actions, b_logprobs, b_advantages, b_returns, action_dim, key):
            # Unreplicate for single-device case
            agent_state_single = flax.jax_utils.unreplicate(agent_state)
            agent_state_updated, loss, pg_loss, v_loss, entropy_loss, approx_kl, key = single_device_update(
                agent_state_single, b_obs, b_actions, b_logprobs, b_advantages, b_returns, action_dim, key
            )
            # Replicate back for consistency with the rest of the code
            return flax.jax_utils.replicate(agent_state_updated, devices=learner_devices), loss, pg_loss, v_loss, entropy_loss, approx_kl, key
    else:
        multi_device_update = jax.pmap(
            single_device_update,
            axis_name="local_devices",
            devices=global_learner_decices,
            in_axes=(0, 0, 0, 0, 0, 0, None, None),
            out_axes=(0, 0, 0, 0, 0, 0, None),
            static_broadcasted_argnums=(6),
        )

    rollout_queue = queue.Queue(maxsize=1)
    params_queues = []
    for d_idx, d_id in enumerate(args.actor_device_ids):
        params_queue = queue.Queue(maxsize=1)
        params_queue.put(jax.device_put(flax.jax_utils.unreplicate(agent_state.params), local_devices[d_id]))
        threading.Thread(
            target=rollout,
            args=(
                jax.device_put(key, local_devices[d_id]),
                args,
                rollout_queue,
                params_queue,
                writer,
                learner_devices,
            ),
        ).start()
        params_queues.append(params_queue)

    rollout_queue_get_time = deque(maxlen=10)
    data_transfer_time = deque(maxlen=10)
    learner_policy_version = 0
    prepare_data = jax.jit(prepare_data, device=learner_devices[0])
    while True:
        learner_policy_version += 1
        if learner_policy_version == 1 or not args.test_actor_learner_throughput:
            rollout_queue_get_time_start = time.time()
            (
                global_step,
                actor_policy_version,
                update,
                obs,
                dones,
                values,
                actions,
                logprobs,
                rewards,
            ) = rollout_queue.get()
            rollout_queue_get_time.append(time.time() - rollout_queue_get_time_start)
            writer.add_scalar("stats/rollout_queue_get_time", np.mean(rollout_queue_get_time), global_step)

        data_transfer_time_start = time.time()
        b_obs, b_actions, b_logprobs, b_advantages, b_returns = prepare_data(
            obs,
            dones,
            values,
            actions,
            logprobs,
            rewards,
            args.gamma,
            args.gae_lambda,
        )
        data_transfer_time.append(time.time() - data_transfer_time_start)
        writer.add_scalar("stats/data_transfer_time", np.mean(data_transfer_time), global_step)

        training_time_start = time.time()
        (agent_state, loss, pg_loss, v_loss, entropy_loss, approx_kl, key) = multi_device_update(
            agent_state,
            b_obs,
            b_actions,
            b_logprobs,
            b_advantages,
            b_returns,
            envs.single_action_space.n,
            key,
        )
        if learner_policy_version == 1 or not args.test_actor_learner_throughput:
            for d_idx, d_id in enumerate(args.actor_device_ids):
                params_queues[d_idx].put(jax.device_put(flax.jax_utils.unreplicate(agent_state.params), local_devices[d_id]))
        if args.profile:
            v_loss[-1, -1, -1].block_until_ready()
        writer.add_scalar("stats/training_time", time.time() - training_time_start, global_step)
        writer.add_scalar("stats/rollout_queue_size", rollout_queue.qsize(), global_step)
        writer.add_scalar("stats/params_queue_size", params_queue.qsize(), global_step)
        print(
            global_step,
            f"actor_policy_version={actor_policy_version}, actor_update={update}, learner_policy_version={learner_policy_version}, training time: {time.time() - training_time_start}s",
        )

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", agent_state.opt_state[1].hyperparams["learning_rate"][0].item(), global_step)
        writer.add_scalar("losses/value_loss", v_loss[-1, -1, -1].item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss[-1, -1, -1].item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss[-1, -1, -1].item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl[-1, -1, -1].item(), global_step)
        writer.add_scalar("losses/loss", loss[-1, -1, -1].item(), global_step)
        if update >= args.num_updates:
            break

    if args.save_model and args.local_rank == 0:
        if args.distributed:
            jax.distributed.shutdown()
        agent_state = flax.jax_utils.unreplicate(agent_state)
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        with open(model_path, "wb") as f:
            f.write(
                flax.serialization.to_bytes(
                    [
                        vars(args),
                        [
                            agent_state.params["network_params"],
                            agent_state.params["actor_params"],
                            agent_state.params["critic_params"],
                        ],
                    ]
                )
            )
        print(f"model saved to {model_path}")

        # Deterministic inline evaluator — argmax policy, raw (unclipped) rewards
        eval_envs = make_env(args.env_id, args.seed + 100, 1)()
        eval_obs, _ = eval_envs.reset()
        eval_obs = np.asarray(eval_obs)  # materialise LazyFrames
        eval_episode_return = 0.0
        eval_episode_length = 0
        episodic_returns = []
        episodic_lengths = []
        while len(episodic_returns) < 10:
            eval_action = get_action_deterministic(
                agent_state.params, eval_obs, eval_envs.single_action_space.n
            )
            eval_obs, eval_reward, eval_terminated, eval_truncated, _ = eval_envs.step(np.array(eval_action))
            eval_obs = np.asarray(eval_obs)  # materialise LazyFrames after each step
            eval_episode_return += float(eval_reward[0])  # raw reward — real game score
            eval_episode_length += 1
            if eval_terminated[0] or eval_truncated[0]:
                episodic_returns.append(eval_episode_return)
                episodic_lengths.append(eval_episode_length)
                eval_episode_return = 0.0
                eval_episode_length = 0
                eval_obs, _ = eval_envs.reset()
                eval_obs = np.asarray(eval_obs)
        eval_envs.close()

        print(f"Evaluation over {len(episodic_returns)} episodes (deterministic policy, raw rewards):")
        print(f"  Return — mean: {np.mean(episodic_returns):.2f}, std: {np.std(episodic_returns):.2f}, "
              f"min: {np.min(episodic_returns):.2f}, max: {np.max(episodic_returns):.2f}")
        print(f"  Length — mean: {np.mean(episodic_lengths):.1f}")
        for idx, (ep_ret, ep_len) in enumerate(zip(episodic_returns, episodic_lengths)):
            writer.add_scalar("eval/episodic_return", ep_ret, idx)
            writer.add_scalar("eval/episodic_length", ep_len, idx)

        if args.upload_model:
            from cleanrl_utils.huggingface import push_to_hub

            repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
            repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
            push_to_hub(
                args,
                episodic_returns,
                repo_id,
                "PPO",
                f"runs/{run_name}",
                f"videos/{run_name}-eval",
                extra_dependencies=["jax", "gymnasium", "atari"],
            )

    envs.close()
    writer.close()
