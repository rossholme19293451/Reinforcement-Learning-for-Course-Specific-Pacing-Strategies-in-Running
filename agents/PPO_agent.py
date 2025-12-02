import torch
import torch.nn as nn
import numpy as np
from pyparsing import actions


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        self.policy_head = nn.Linear(256, action_dim)
        self.value_head = nn.Linear(256, 1)

        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, x):
        h = self.net(x)
        return self.policy_head(h), self.value_head(h)

    def get_action(self, obs):
        mean, value = self.forward(obs)
        std = self.log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob, value

class PPO_Agent:
    def __init__(
            self,
            env_fn,
            frames_per_batch = 1024,
            total_frames = 50_000,
            gamma = 0.99,
            lam = 0.95,
            clip_epsilon = 0.2,
            lr = 0.01,
            epochs = 10,
            minibatch_size = 64,
            device = 'cpu',
    ):
        self.env = env_fn
        self.frames_per_batch = frames_per_batch
        self.total_frames = total_frames
        self.gamma = gamma
        self.lam = lam
        self.clip_epsilon = clip_epsilon
        self.lr = lr
        self.epochs = epochs
        self.minibatch_size = minibatch_size
        self.device = device

        obs_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]

        self.model = ActorCritic(obs_dim, action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def collect_batch(self):
        obs_list = []
        actions = []
        log_probs = []
        rewards = []
        dones = []
        values = []

        obs, _ = self.env.reset()
        for _ in range(self.frames_per_batch):
            obs_t = torch.tensor(obs).to(self.device)
            with torch.no_grad():
                action, log_prob, value = self.model.get_action(obs_t)

            action = action.cpu().numpy()
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            obs_list.append(obs)
            actions.append(action)
            log_probs.append(log_prob.cpu().numpy())
            rewards.append(reward)
            dones.append(done)
            values.append(value.cpu().numpy())

            obs = next_obs

            if done:
                obs, _ = self.env.reset()

        return (np.array(obs_list),
                np.array(actions),
                np.array(log_probs),
                np.array(rewards),
                np.array(dones),
                np.array(values).reshape(-1),
                )

    def compute_gae(self, rewards, values, dones):
        adv = np.zeros_like(rewards)
        last_gae_lam = 0

        for t in reversed(range(len(rewards))):
            next_non_terminal = 1.0 - dones[t]
            next_value = values[t + 1] if t + 1 < len(rewards) else 0
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            adv[t] = last_gae_lam = delta + self.gamma * self.lam * next_non_terminal * last_gae_lam

        returns = adv + values
        return adv, returns

    def train(self):
        frame_count = 0
        while frame_count < self.total_frames:
            (
                obs_list,
                actions,
                old_log_probs,
                rewards,
                dones,
                values,
            ) = self.collect_batch()

            advantages, returns = self.compute_gae(rewards, values, dones)

            advantages = (advantages - advantages.mean()) / (advantages.std() + 0.00001)

            dataset_size = len(obs_list)
            for epoch in range(self.epochs):
                idxs = np.arange(dataset_size)
                np.random.shuffle(idxs)

                for start in range(0, dataset_size, self.minibatch_size):
                    end = start + self.minibatch_size
                    batch_idx = idxs[start:end]

                    obs_t = torch.tensor(obs_list[batch_idx]).to(self.device)
                    actions_t = torch.tensor(actions[batch_idx]).to(self.device)
                    old_log_probs_t = torch.tensor(old_log_probs[batch_idx]).to(self.device)
                    adv_t = torch.tensor(advantages[batch_idx]).to(self.device)
                    ret_t = torch.tensor(returns[batch_idx]).to(self.device)

                    mean, value = self.model(obs_t)
                    std = self.model.log_std.exp()
                    dist = torch.distributions.Normal(mean, std)
                    new_log_prob = dist.log_prob(actions_t).sum(-1)

                    ratio = (new_log_prob - old_log_probs_t).exp()

                    surr1 = ratio * adv_t
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * adv_t
                    policy_loss = -torch.min(surr1, surr2).mean()

                    value_loss = (ret_t - value.squeeze()).pow(2).mean()

                    entropy = dist.entropy().sum(-1).mean()

                    loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                frame_count += self.frames_per_batch
                print(f"Frames: {frame_count}/{self.total_frames}")

    def run(self, env_params=None, episodes = 1):
        if env_params is None:
            self.env = env_params

        all_episodes = []

        for ep in range(episodes):
            obs, _ = self.env.reset()
            done = False
            total_reward = 0

            episode_data = {
                "distance": [],
                "velocity": [],
                "energy": [],
                "action": [],
                "reward": [],
                "time": []
            }

            while not done:
                obs_t = torch.tensor(obs).to(self.device)
                with torch.no_grad():
                    mean, _ = self.model(obs_t.unsqueeze(0))
                action = mean.cpu().numpy()[0]

                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                total_reward += reward

                episode_data["distance"].append(obs[0])
                episode_data["velocity"].append(obs[1])
                episode_data["energy"].append(obs[2])
                episode_data["action"].append(action)
                episode_data["reward"].append(reward)
                episode_data["time"].append(info["time"])

            all_episodes.append(episode_data)
            print(f"Episode: {ep}, Reward: {total_reward}")

        return all_episodes




