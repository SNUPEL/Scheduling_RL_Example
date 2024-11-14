import os
import json

class Configuration:
    def __init__(self):
        self.load_model = False
        self.save_model = True
        self.log_path = None

        self.keyword = "241114_final_check"  # 폴더명

        # Problem related parameters
        self.data = "./Data/data_02.csv"
        self.num_job = 50
        self.num_m = 5

        # Environment related parameters
        self.weight_setup = 0.5
        self.weight_tard = 0.5
        self.debug_mode = False

        # Training
        self.n_episode = 1000
        self.n_record = 1000

        # Agent related hyperparameters
        self.state_dim =  2 * self.num_m + 1 + self.num_job
        self.hidden_dim = 256
        self.action_dim = self.num_job
        self.gamma = 0.99
        self.lmbda = 0.95
        self.eps_clip = 0.2
        self.K_epoch = 1
        self.lr = 1e-4
        self.T_horizon = 1

        self.dirpath = './output/{0}/'.format(self.keyword)
        if not os.path.exists(self.dirpath):
            os.makedirs(self.dirpath)

        self.model_dir = self.dirpath + 'model/'
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        self.log_dir = self.dirpath + 'log/'
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        with open(self.log_dir + "train_log.csv", 'w') as f:
            f.write('episode,reward,reward_tard,reward_setup,training time\n')

        # Save the configuration as a JSON file
        self.save_config_as_json()

    def save_config_as_json(self):
        # Convert class attributes to a dictionary
        config_dict = {key: value for key, value in self.__dict__.items() if not key.startswith('__')}

        # Define the path where the JSON will be saved
        config_json_path = os.path.join(self.dirpath, 'config.json')

        # Save the dictionary as a JSON file
        with open(config_json_path, 'w') as f:
            json.dump(config_dict, f, indent=4)