import os
import torch

from cfg import Configuration
from Modules.env import PMSP
from Modules.ppo import PPO
import copy
import time
device = torch.device("cpu")
print(device)

if __name__ == "__main__":
    cfg = Configuration()

    env = PMSP(cfg)
    agent = PPO(cfg, cfg.state_dim, cfg.hidden_dim, cfg.num_job)
    for episode in range(0, cfg.n_episode):

        # region logging
        if (episode + 1) % cfg.n_record == 0 or episode == 1:
            with open(cfg.log_dir + "train_log_episode_{0}.csv".format(episode), 'w') as f:
                f.write('Action,Job1,P1,Job2,P2,Job3,P3,Job4,P4,reward,States\n')
        if (episode + 1) % cfg.n_record == 0 or episode == 1:
            with open(cfg.log_dir + "train_log_episode_{0}_masked.csv".format(episode), 'w') as f:
                f.write('Action,Job1,P1,Job2,P2,Job3,P3,Job4,P4,reward\n')
        # endregion

        state = env.reset()
        r_epi = 0.0
        start = time.time()
        done = False
        # masking
        done_mask = torch.ones(cfg.num_job)
        action_list = []
        while not done:
            logit = agent.pi(state)
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

            # region logging
            if (episode + 1) % cfg.n_record == 0 or (episode + 1) == 1:
                with open(cfg.log_dir + "train_log_episode_{0}.csv".format(episode), 'a') as f:
                    f.write('{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},'.format(
                        action, indices[0], values[0], indices[1], values[1], indices[2], values[2], indices[3],
                        values[3], reward))
                    dd = ''
                    for s in state:
                        dd += str(s.item()) + ','
                    f.write(dd)
                    f.write('\n')
                with open(cfg.log_dir + "train_log_episode_{0}_masked.csv".format(episode), 'a') as f:
                    f.write('{0},{1},{2},{3},{4},{5},{6},{7},{8},{9}\n'.format(
                        action, masked_indices[0], masked_values[0], masked_indices[1], masked_values[1],
                        masked_indices[2], masked_values[2], masked_indices[3], masked_values[3],
                        reward))
            # endregion

            agent.put_data((state, action, reward, next_state, prob[action].item(), done))
            state = next_state

            r_epi += reward
            if done:
                tardiness = env.reward_tard / env.num_job
                setup = env.reward_setup / env.num_job
                makespan = env.makespan
                finish = time.time()
                if (episode + 1) % cfg.n_record == 0 or (episode + 1) == 1:
                    print("episode: %d | reward: %.4f | Setup: %.4f | Tardiness %.4f | Makespan %.4f | Training Time %.4f " % (
                        episode, r_epi, setup, tardiness, makespan, round(finish - start, 4)))
                    model_save_path = os.path.join(cfg.model_dir, f"episode_{episode}.pt")
                    torch.save(agent.state_dict(), model_save_path)

                with open(cfg.log_dir + "train_log.csv", 'a') as f:
                    f.write('%d,%.2f,%.2f,%.2f,%.2f\n' % (
                        episode, r_epi, env.reward_tard, env.reward_setup, round(finish - start, 4)))

                break
        agent.train_net()




