# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_procgenpy
import argparse
import os
import random
import time
from distutils.util import strtobool

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

import powderworld.default_envs
from powderworld.sim import pw_elements

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
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="Powderworld",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")

    # Env settings
    parser.add_argument("--pw-env", type=str, default="PWEnvSandPlace",
        help="the id of the environment")
    parser.add_argument("--flatten-actions", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, flatten multidiscrete action space")
    
    # Algorithm specific arguments
    parser.add_argument("--total-timesteps", type=int, default=int(1e6),
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=64,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=256,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.999,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=8,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=3,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.01,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


# taken from https://github.com/AIcrowd/neurips2020-procgen-starter-kit/blob/142d09586d2272a17f44481a115c4bd817cf6a94/models/impala_cnn_torch.py
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)

    def forward(self, x):
        inputs = x
        x = nn.functional.relu(x)
        x = self.conv0(x)
        x = nn.functional.relu(x)
        x = self.conv1(x)
        return x + inputs


class ConvSequence(nn.Module):
    def __init__(self, input_shape, out_channels):
        super().__init__()
        self._input_shape = input_shape
        self._out_channels = out_channels
        self.conv = nn.Conv2d(in_channels=self._input_shape[0], out_channels=self._out_channels, kernel_size=3, padding=1)
        self.res_block0 = ResidualBlock(self._out_channels)
        self.res_block1 = ResidualBlock(self._out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = nn.functional.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.res_block0(x)
        x = self.res_block1(x)
        assert x.shape[1:] == self.get_output_shape()
        return x

    def get_output_shape(self):
        _c, h, w = self._input_shape
        return (self._out_channels, (h + 1) // 2, (w + 1) // 2)


class Agent(nn.Module):
    def __init__(self, envs, args):
        super().__init__()
        h, w = envs.single_observation_space.shape
        self.element_embed = nn.Embedding(num_embeddings=len(pw_elements), embedding_dim=32)
        shape = (32, h, w)
        conv_seqs = []
        for out_channels in [16, 32, 32]:
            conv_seq = ConvSequence(shape, out_channels)
            shape = conv_seq.get_output_shape()
            conv_seqs.append(conv_seq)
        conv_seqs += [
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(in_features=shape[0] * shape[1] * shape[2], out_features=256),
            nn.ReLU(),
        ]
        self.network = nn.Sequential(*conv_seqs)
        if args.flatten_actions:
            print("Initializing action head w size {}.".format(envs.single_action_space.n))
            self.actor = layer_init(nn.Linear(256, envs.single_action_space.n), std=0.01)
        else:
            self.actors = nn.ModuleList()
            self.action_embeds = nn.ModuleList()
            for ac_dim in range(len(envs.single_action_space)):
                ac_num = envs.single_action_space[ac_dim].n
                
                # Each action head takes 256 feature vec + 16 feature vec for each previous action.
                self.actors.append(nn.Sequential(
                    nn.Linear(256 + 16*ac_dim, 256),
                    nn.ReLU(),
                    layer_init(nn.Linear(256, ac_num), std=0.01),
                ))
                
                # Embed each action taken into 16-length vector.
                self.action_embeds.append(nn.Embedding(num_embeddings=ac_num, embedding_dim=16))
                print("Initializing action head w size {}.".format(envs.single_action_space[ac_dim].n))
            
        self.critic = layer_init(nn.Linear(256, 1), std=1)
        
    def parse_obs(self, x):
        # "bhw" -> "bchw"
        x = self.element_embed(x.int())
        x = x.permute((0,3,1,2))
        return self.network(x)

    def get_value(self, x):
        return self.critic(self.parse_obs(x))

    def get_action_and_value(self, x, action=None, disabled_elements=None):
        hidden = self.parse_obs(x)
        if args.flatten_actions:
            logits = self.actor(hidden)
            probs = Categorical(logits=logits)
            if action is None:
                action = probs.sample()
            return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)
        else:
            action_stack = []
            logprob_stack = []
            entropy = 0
            value = self.critic(hidden)
            for i, actor in enumerate(self.actors):
                logits = actor(hidden)
                
                if i == 0:
                    if disabled_elements is not None:
                        for n, disabled_for_task in enumerate(disabled_elements):
                            for elem in disabled_for_task:
                                logits[n, elem] = -np.inf
                probs = Categorical(logits=logits)
                    
                if action is None:
                    action_sample = probs.sample()
                else:
                    action_sample = action[:, i]
                action_stack.append(action_sample)
                logprob_stack.append(probs.log_prob(action_sample))
                entropy += probs.entropy()
                
                hidden = torch.concat([hidden, self.action_embeds[i](action_sample)], dim=1)
                
            return torch.stack(action_stack, -1), torch.stack(logprob_stack, -1), entropy, value


if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.pw_env}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
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
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print("Using device,", device)
    args.device = device

    # env setup
    env_fnc = getattr(powderworld.default_envs, args.pw_env)
    envs = env_fnc(num_envs=args.num_envs, device=device, use_jit=True, flatten_actions=args.flatten_actions)
    envs.single_action_space = envs.action_space
    envs.single_observation_space = envs.observation_space
    envs.is_vector_env = True
    envs = gym.wrappers.RecordEpisodeStatistics(envs)
    if args.capture_video and args.track:
        from powderworld.monitoring.record_video_wrapper import RecordVideoWrapper
        envs = RecordVideoWrapper(envs, f"videos/{run_name}", wandb=wandb)
        # envs = gym.wrappers.RecordVideo(envs, f"videos/{run_name}")
    envs = gym.wrappers.NormalizeReward(envs, gamma=args.gamma)
    envs = gym.wrappers.TransformReward(envs, lambda reward: np.clip(reward, -10, 10))
    if args.flatten_actions:
        assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"
    else:
        assert isinstance(envs.single_action_space, gym.spaces.MultiDiscrete), "only multidiscrete action space is supported"

    agent = Agent(envs, args).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    if args.flatten_actions:
        actions = torch.zeros((args.num_steps, args.num_envs)).to(device)
        logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    else:
        actions = torch.zeros((args.num_steps, args.num_envs, len(envs.single_action_space))).to(device)
        logprobs = torch.zeros((args.num_steps, args.num_envs, len(envs.single_action_space))).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs = torch.Tensor(envs.reset()).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size
    
    rew_normalize_mean = None
    rew_normalize_std = None

    for update in range(1, num_updates + 1):
        start_time_update = time.time()
        env_returns = []
        
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow
            
        ac_times = []
        env_times = []

        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                # [num_envs, action_dim]
                disabled_elements = []
                has_disabled = False
                for task in envs.tasks:
                    dis = task.config['agent']['disabled_elements']
                    disabled_elements.append(dis)
                    if len(dis) > 0:
                        has_disabled = True
                if has_disabled:
                    action, logprob, _, value = agent.get_action_and_value(next_obs, disabled_elements=disabled_elements)
                else:
                    action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, done, info = envs.step(action.cpu().numpy())
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)

            for item in info:
                if "episode" in item.keys():
                    env_returns.append(item["episode"]["r"])
                    writer.add_scalar("charts/episodic_return", item["episode"]["r"], global_step)
                    # writer.add_scalar("charts/episodic_length", item["episode"]["l"], global_step)
                    break
                    
        if update == 1:
            rew_normalize_mean = np.array([env_returns]).mean()
            rew_normalize_std = max(1, np.array([env_returns]).std() + 0.001)
        
        if len(env_returns) > 0:
            env_rew_mean = np.array([env_returns]).mean()
            print(f"global_step={global_step}, avg_return={env_rew_mean}")
            writer.add_scalar("charts/avg_return_empirical", env_rew_mean, global_step)
            writer.add_scalar("charts/avg_return_empirical_normalized", (env_rew_mean - rew_normalize_mean) / rew_normalize_std, global_step)
            # Note: normalized log here is only for debugging. Because the means/std are computed on-the-fly, 
            # you can't compare it between environments or between runs.

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        if args.flatten_actions:
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape(-1)
        else:
            b_logprobs = logprobs.reshape(-1, len(envs.single_action_space))
            b_actions = actions.reshape((-1, len(envs.single_action_space)))
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        
        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                if not args.flatten_actions:
                    mb_advantages = mb_advantages[:, None]
                # Advantages = [batch, 1]
                # Ratio = [batch, num_ac_dim]
                # Increase action probability for each sampled dimension, by advantage.

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        
        # writer.add_scalar("charts/avg_reward", rewards.mean().item(), global_step)
        # writer.add_scalar("charts/avg_value_pred", values.mean().item(), global_step)
        # writer.add_scalar("charts/avg_advantage", advantages.mean().item(), global_step)
        # writer.add_scalar("charts/avg_return_gae", returns.mean().item(), global_step)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        print(pg_loss.item())
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int((args.num_steps * args.num_envs) / (time.time() - start_time_update)))
        writer.add_scalar("charts/SPS", int((args.num_steps * args.num_envs) / (time.time() - start_time_update)), global_step)

    envs.close()
    writer.close()