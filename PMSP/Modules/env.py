import numpy as np
import torch
import pandas as pd
import copy

class Job:
    def __init__(self, index = None,
                 name = None,
                 created = None,
                 processing_time=0,
                 due_date = None,
                 feature=0):
        self.name = name
        self.index = index
        self.created = created
        self.processing_time = processing_time
        self.due_date = due_date
        self.feature = feature

        self.is_tardy = None
        self.machine = None
        self.setup_time = None
        self.expected_tard = None
        self.completion_time = 0
        self.done = False

    def calculate_expected_tard(self, expected_completion):
        self.expected_tard = max(expected_completion-self.due_date, 0)
        return self.expected_tard

class PMSP:
    def __init__(self, cfg, debug_mode = None, data = None):
        self.num_job = cfg.num_job  # scheduling 대상 job의 수
        self.num_m = cfg.num_m  # parallel machine 수
        self.weight_setup = cfg.weight_setup
        self.weight_tard =cfg.weight_tard
        self.debug_mode = cfg.debug_mode if debug_mode is None else debug_mode
        self.data = pd.read_csv(cfg.data) if data is None else pd.read_csv(data)
        self.state_dim = cfg.state_dim
        
        self.now = 0.0
        self.makespan = 0.0
        self.r_t = 0.0
        self.r_s = 0.0

        self.machine = None
        self.done = False
        self.previous_time_step = 0
        self.reward_setup = 0
        self.reward_tard = 0
        self.num_scheduled = 0

        # Job 객체 생성 리스트
        self.job_list = self.generate_jobs()

        self.machine_ESD = [0.0 for i in range(self.num_m)]
        self.machine_started = [-1.0 for i in range(self.num_m)]
        self.machine_setup = [0 for i in range(self.num_m)]

        # Debugging variables
        self.done_mask = torch.ones(self.num_job)
        self.f4 = torch.zeros(self.num_job)
        for i in range(self.num_job):
            self.f4[i] = self.job_list[i].feature

    def generate_jobs(self):
        job_list = list()
        # DataFrame의 각 행을 순회하며 Job 객체 생성
        for _, row in self.data.head(self.num_job).iterrows():
            job = Job(
                index=row['idx'],
                name=row['name'],
                created=row['created'],
                processing_time=row['processing_time'],
                due_date=row['due_date'],
                feature=row['feature']
            )
            job_list.append(job)
        return job_list

    def step(self, action):
        if self.debug_mode:
            print(str(round(self.now,2)),'PMSP got action ',action)

        done = False
        earliest_machine = self.machine_ESD.index(min(self.machine_ESD))
        self.machine = earliest_machine
        job = next((x for x in self.job_list if x.name == 'Job ' + str(action)), None)
        job.done = True
        self.num_scheduled +=1
        original_setup = copy.deepcopy(self.machine_setup[earliest_machine])
        setup_time = abs(original_setup - job.feature)
        # print("Original setup이 {0} 이고 Job feature가 {1}이므로 setup time은 {2}입니다.".format(original_setup, job.feature, setup_time))
        self.machine_setup[earliest_machine] = job.feature
        self.machine_started[earliest_machine] = self.now
        self.machine_ESD[earliest_machine] += (setup_time + job.processing_time)

        expected_completion = copy.deepcopy(self.machine_ESD[earliest_machine])
        expected_tardiness = job.calculate_expected_tard(expected_completion)
        r_s, r_t, reward = self._calculate_reward(setup_time, expected_tardiness)

        if self.debug_mode:
            print(str(round(self.now,2)), '(PMSP)','Job',action, '의 expected tardiness는', str(round(expected_tardiness,2)), '입니다.')
        self.r_t = r_t
        self.r_s = r_s
        if self.debug_mode:
            print(str(round(self.now,2)),'Job ',action,'을 선택했을 때 받은 reward는 ',reward,'입니다.')

        self.done_mask[job.index] = 0.0
        next_state = self._get_state()
        # update masking
        if self.num_scheduled == self.num_job:
            done = True
            self.makespan = max(self.machine_ESD)


        # move to next time event
        next_routing = min(self.machine_ESD)
        if self.debug_mode:
            print(str(round(self.now,2)),"\t다음 시간 간격으로 이동합니다.")
        self.now = next_routing
        if self.debug_mode:
            print(str(round(self.now,2)),"\t다음 시간 간격으로 이동하여 시간이 {0}이 되었습니다.".format(self.now))

        return next_state, reward, done

    def reset(self):
        self.now = 0.0
        self.r_s = 0.0
        self.r_t = 0.0
        self.done = False
        self.reward_setup = 0
        self.reward_tard = 0
        self.done_mask = torch.ones(self.num_job)
        self.job_list = self.generate_jobs()

        self.machine_ESD = [0.0 for i in range(self.num_m)]
        self.machine_started = [-1.0 for i in range(self.num_m)]
        self.machine_setup = [0 for i in range(self.num_m)]
        self.makespan = 0.0
        self.num_scheduled = 0
        self.f4 = torch.zeros(self.num_job)
        for i in range(self.num_job):
            self.f4[i] = self.job_list[i].feature

        return self._get_state()


    def _get_state(self):
        f_1 = torch.tensor([self.machine], dtype=torch.float) if self.machine is not None else torch.tensor([-1.0], dtype=torch.float)
        f_2 = torch.zeros(self.num_m, dtype=torch.float)  # Setup -> 현재 라인의 셋업 값과 같은 셋업인 job의 수
        f_3 = torch.zeros(self.num_m, dtype=torch.float)  # General Info -> 각 라인의 progress rate
        f_4 = self.f4.clone().detach()

        for i in range(len(self.done_mask)):
            if self.done_mask[i] == 0:
                f_4[i] = -1

        remaining_jobs = list()
        for j in self.job_list:
            if not j.done:
                remaining_jobs.append(j)

        for line_num in range(self.num_m):
            line_setup = self.machine_setup[line_num]
            same_setup_list = [1 for job in remaining_jobs if job.feature == line_setup]
            f_2[line_num] = np.sum(same_setup_list) / len(remaining_jobs) if len(remaining_jobs) > 0 else 0.0

            if self.machine_ESD[line_num] > self.now:  # 현재 작업 중인 job이 있을 때
                f_3[line_num] = (self.now - self.machine_started[line_num]) / (
                            self.machine_ESD[line_num] - self.machine_started[line_num])
                # print(str(round(self.now,2)),'\t Machine',str(line_num),'은 지금 작업중이며, 진행률은 ',f_3[line_num],'입니다.')

        state = torch.cat((f_1, f_2, f_3, f_4),dim=0)
        return state

    def _calculate_reward(self, setup, tard):
        reward_1= (1 - setup / 5) # setup이 없을수록 커서 좋은 reward를 의미함
        reward_2 = np.exp(-tard/10) # tardiness가 0이면 1, 큰 값일수록 0에 가까워짐.
        self.reward_setup += reward_1
        self.reward_tard += reward_2

        if self.debug_mode:
            print('Reward_setup \t\t:',reward_1,"(Cumulative sum:",round(self.reward_setup,3),")")
            print('Reward_tardiness \t:',round(reward_2,3),"(Cumulative sum:",round(self.reward_tard,3),")")

        reward = reward_1 * self.weight_setup + reward_2 * self.weight_tard
        return reward_1, reward_2, reward




