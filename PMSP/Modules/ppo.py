import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class PPO(nn.Module):
    def __init__(self, cfg, state_dim, hidden_dim, action_dim):
        super(PPO, self).__init__()
        self.gamma = cfg.gamma
        self.lmbda = cfg.lmbda
        self.eps_clip = cfg.eps_clip
        self.K_epoch = cfg.K_epoch
        self.device = torch.device("cpu")
        self.lr = cfg.lr
        self.data = []

        self.fc1 = nn.Linear(state_dim, hidden_dim).to(self.device)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim).to(self.device)
        self.fc_pi = nn.Linear(hidden_dim, action_dim).to(self.device)
        self.fc_v = nn.Linear(hidden_dim, 1).to(self.device)
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

    # define policy iteration network
    def pi(self, x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc_pi(x)
        return x.to(self.device)

    # define value network
    def v(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        v = self.fc_v(x)
        return v.to(self.device)

    def put_data(self, transition):
        self.data.append(transition)

    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, prob_a, done = transition

            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            done_mask = 0 if done else 1
            done_lst.append([done_mask])

        s=torch.stack(s_lst, dim=0).to(self.device)
        a=torch.tensor(a_lst, dtype=torch.int64).to(self.device)
        r=torch.tensor(r_lst, dtype=torch.float).to(self.device)
        s_prime = torch.stack(s_prime_lst, dim=0).to(self.device)
        prob_a = torch.tensor(prob_a_lst, dtype=torch.float).to(self.device)
        done = torch.tensor(done_lst, dtype=torch.float).to(self.device)

        self.data = []
        return s, a, r, s_prime, prob_a, done

    def train_net(self):
        s, a, r, s_prime, prob_a, done = self.make_batch()

        for i in range(self.K_epoch):
            td_target = r + self.gamma * self.v(s_prime) * done
            delta = td_target - self.v(s)
            delta = delta.cpu().detach().numpy()

            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = self.gamma * self.lmbda * advantage + delta_t[0]

                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float).to(self.device)

            # normalizing to ensure stability
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

            logit = self.pi(s)
            pi = torch.softmax(logit, dim=-1)
            pi_a = pi.gather(1, a)
            ratio = torch.log(pi_a) - torch.log(prob_a)  # a/b == exp(log(a)-log(b))

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage
            loss = -torch.min(surr1, surr2) + 0.5 * F.smooth_l1_loss(self.v(s), td_target.detach())
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

    def save(self, episode, file_dir):
        torch.save({"episode": episode,
                    "model_state_dict": self.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict()},
                   file_dir + "episode-%d.pt" % episode)

