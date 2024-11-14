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
               data = "Data/data_01.csv")
    agent = PPO(cfg, cfg.state_dim, cfg.hidden_dim, cfg.num_job)

    # Define the model path
    model_path = "output/241113_data_01_002(Adam)/model/episode_21200.pt"

    # Load the saved model
    agent.load_state_dict(torch.load(model_path))

    # Make sure to call .eval() if you want to set the agent in evaluation mode
    agent.eval()
    state = env.reset()
    r_epi = 0.0
    done = False

    # masking
    done_mask = torch.ones(cfg.num_job)
    action_list = []
    while not done:
        logit = agent.pi(torch.from_numpy(state).float())
        prob = torch.softmax(logit, dim=-1)
        masked_prob = prob.mul(done_mask)
        sum = torch.sum(masked_prob, dim=-1)
        masked_prob = masked_prob.div(sum)
        p = masked_prob.tolist()

        # 가장 큰 값 몇개를 뽑아서 보고 싶을 때
        values, indices = torch.sort(prob, descending=True)
        masked_values, masked_indices = torch.sort(masked_prob, descending=True)
        m = torch.distributions.Categorical(probs=masked_prob)
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




