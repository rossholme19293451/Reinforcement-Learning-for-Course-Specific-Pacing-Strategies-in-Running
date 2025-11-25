import torch
import torch.nn as nn

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
            lr = 0.001,
            epochs = 10,
            minibatch_size = 64,
            device = 'cpu',
    ):
        self.env_fn = env_fn
        self.frames_per_batch = frames_per_batch
        self.total_frames = total_frames
        self.gamma = gamma
        self.lam = lam
        self.clip_epsilon = clip_epsilon
        self.lr = lr
        self.epochs = epochs
        self.minibatch_size = minibatch_size
        self.device = device

        obs_dim = self.env_fn.observation_space.shape[0]
        action_dim = self.env_fn.action_space.shape[0]

        self.model = ActorCritic(obs_dim, action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)



