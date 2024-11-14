import torch

from cfg import Configuration
from Modules.env import PMSP
from Modules.ppo import PPO
import copy
device = torch.device("cpu")
print(device)
if __name__ == "__main__":
    cfg = Configuration()

    env = PMSP(cfg,
               debug_mode = True,
               data = "Data/data_03.csv")

    state = env.reset()
    r_epi = 0.0
    done = False

    # masking
    done_mask = torch.ones(cfg.num_job)
    action_list = []

    while not done:
        prob = done_mask.clone().detach() # 남아 있는 작업을 대상으로 Random Sampling
        m = torch.distributions.Categorical(probs=prob)
        action = m.sample().item()
        action_list.append(action)
        done_mask[action] = 0.0
        next_state, reward, done = env.step(action)
        reward_copy = copy.deepcopy(reward)
        state = next_state
        r_epi += reward
        if done:
            tardiness = env.reward_tard / env.num_job
            setup = env.reward_setup / env.num_job
            makespan = env.makespan
            print("reward: %.4f | Setup: %.4f | Tardiness %.4f | makespan %.4f | Loss %.4f " % (
                r_epi, setup, tardiness, makespan, 0.0))
            break




