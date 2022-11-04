import numpy as np

from .base_policy import BasePolicy


class MPCPolicy(BasePolicy):

    def __init__(self,
                 env,
                 ac_dim,
                 dyn_models,
                 horizon,
                 N,
                 sample_strategy='random',
                 cem_iterations=4,
                 cem_num_elites=5,
                 cem_alpha=1,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        # init vars
        self.env = env
        self.dyn_models = dyn_models
        self.horizon = horizon
        self.N = N
        self.data_statistics = None  # NOTE must be updated from elsewhere

        self.ob_dim = self.env.observation_space.shape[0]

        # action space
        self.ac_space = self.env.action_space
        self.ac_dim = ac_dim
        self.low = self.ac_space.low
        self.high = self.ac_space.high

        # Sampling strategy
        allowed_sampling = ('random', 'cem')
        assert sample_strategy in allowed_sampling, f"sample_strategy must be one of the following: {allowed_sampling}"
        self.sample_strategy = sample_strategy
        self.cem_iterations = cem_iterations
        self.cem_num_elites = cem_num_elites
        self.cem_alpha = cem_alpha

        print(f"Using action sampling strategy: {self.sample_strategy}")
        if self.sample_strategy == 'cem':
            print(f"CEM params: alpha={self.cem_alpha}, "
                + f"num_elites={self.cem_num_elites}, iterations={self.cem_iterations}")

    def sample_action_sequences(self, num_sequences, horizon, obs=None):
        if self.sample_strategy == 'random' \
            or (self.sample_strategy == 'cem' and obs is None):
            # TODO(Q1) uniformly sample trajectories and return an array of
            # dimensions (num_sequences, horizon, self.ac_dim) in the range
            # [self.low, self.high]
            ra_seq = []

            for _ in range(num_sequences):
                one_seq = []
                for _ in range(horizon):
                    ac = self.ac_space.sample()
                    one_seq.append(ac)
                ra_seq.append(one_seq)

            random_action_sequences = np.clip(ra_seq, self.low, self.high)
            return random_action_sequences
        elif self.sample_strategy == 'cem':
            # TODO(Q5): Implement action selection using CEM.
            # Begin with randomly selected actions, then refine the sampling distribution
            # iteratively as described in Section 3.3, "Iterative Random-Shooting with Refinement" of
            # https://arxiv.org/pdf/1909.11652.pdf 
            for i in range(self.cem_iterations):
                # - Sample candidate sequences from a Gaussian with the current 
                #   elite mean and variance
                #     (Hint: remember that for the first iteration, we instead sample
                #      uniformly at random just like we do for random-shooting)
                # - Get the top `self.cem_num_elites` elites
                #     (Hint: what existing function can we use to compute rewards for
                #      our candidate sequences in order to rank them?)
                # - Update the elite mean and variance
                alpha = self.cem_alpha
                D_acs = self.ac_dim

                if i == 0:
                    ra_seq = []
                    for _ in range(num_sequences):
                        one_seq = []
                        for _ in range(horizon):
                            ac = self.ac_space.sample()
                            one_seq.append(ac)
                        ra_seq.append(one_seq)
                # else:
                #     ra_seq = []
                #     for t in range(horizon):
                #         # print('elites_mean.shape', elites_mean.shape)
                #         # print('num_sequences', num_sequences)
                #         # print('horizon', horizon)
                #         elites_mean_t = elites_mean[t, :]
                #         elites_var_t = elites_var[t, :]
                #         one_seq = np.random.multivariate_normal(mean = elites_mean_t, 
                #             cov = np.diag(elites_var_t), size = num_sequences)
                #         ra_seq.append(one_seq)
                #     ra_seq = np.array(ra_seq).reshape((num_sequences, horizon, D_acs))

                # random_action_sequences = np.clip(ra_seq, self.low, self.high)
                # D_acs = random_action_sequences.shape[2]
                # # print('random_action_sequences.shape', random_action_sequences.shape)

                # res_mean = self.evaluate_candidate_sequences(random_action_sequences, obs)
                # J = self.cem_num_elites
                # idx = np.argpartition(res_mean, -J)[-J:]
                # # idx = idx[np.argsort(res_mean[idx])]
                # elites = random_action_sequences[idx]
                # mean_A = np.mean(elites, axis = 0)
                # var_A = np.var(elites, axis = 0)

                # if i == 0:
                #     elites_mean = mean_A 
                #     elites_var = var_A 
                # else:
                #     elites_mean = alpha * mean_A + (1-alpha) * elites_mean
                #     elites_var = alpha * var_A + (1-alpha) * elites_var
                else:
                    # ra_seq = []
                    # for t in range(horizon):
                    #     # print('elites_mean.shape', elites_mean.shape)
                    #     # print('num_sequences', num_sequences)
                    #     # print('horizon', horizon)
                    #     elites_mean_t = elites_mean[t, :]
                    #     elites_sd_t = elites_sd[t, :]
                    #     one_seq = np.random.normal(elites_mean_t, elites_sd_t, num_sequences)
                    #     ra_seq.append(one_seq)
                    # print('elites_mean.shape', elites_mean.shape)
                    # print('elites_sd.shape', elites_sd.shape)
                    # ra_seq = np.array(ra_seq).reshape((num_sequences, horizon, D_acs))
                    ra_seq = np.random.normal(elites_mean, elites_sd, (num_sequences, horizon, D_acs))

                random_action_sequences = np.clip(ra_seq, self.low, self.high)
                
                # print('random_action_sequences.shape', random_action_sequences.shape)

                res_mean = self.evaluate_candidate_sequences(random_action_sequences, obs)
                J = self.cem_num_elites
                idx = np.argpartition(res_mean, -J)[-J:]
                # idx = idx[np.argsort(res_mean[idx])]
                elites = random_action_sequences[idx]
                mean_A = np.mean(elites, axis = 0)
                sd_A = np.std(elites, axis = 0)

                if i == 0:
                    elites_mean = mean_A 
                    elites_sd = sd_A 
                else:
                    elites_mean = alpha * mean_A + (1-alpha) * elites_mean
                    elites_sd = alpha * sd_A + (1-alpha) * elites_sd

            # TODO(Q5): Set `cem_action` to the appropriate action chosen by CEM
            cem_action = elites_mean

            return cem_action[None]
        else:
            raise Exception(f"Invalid sample_strategy: {self.sample_strategy}")

    def evaluate_candidate_sequences(self, candidate_action_sequences, obs):
        # TODO(Q2): for each model in ensemble, compute the predicted sum of rewards
        # for each candidate action sequence.
        #
        # Then, return the mean predictions across all ensembles.
        # Hint: the return value should be an array of shape (N,)

        # print("obs: ", obs.shape)
        # print("acs: ", candidate_action_sequences.shape)
        res = []
        for model in self.dyn_models:
            res_one = self.calculate_sum_of_rewards(obs, candidate_action_sequences, model)
            res.append(res_one)

        res_mean = np.mean(res, axis = 0)

        # N = candidate_action_sequences.shape[0]
        # assert res_mean.shape == (N, )
        # print("res_np: ", res_np,shape)
        return res_mean

    def get_action(self, obs):
        if self.data_statistics is None:
            return self.sample_action_sequences(num_sequences=1, horizon=1)[0]

        # sample random actions (N x horizon x action_dim)
        candidate_action_sequences = self.sample_action_sequences(
            num_sequences=self.N, horizon=self.horizon, obs=obs)

        if candidate_action_sequences.shape[0] == 1:
            # CEM: only a single action sequence to consider; return the first action
            return candidate_action_sequences[0][0][None]
        else:
            predicted_rewards = self.evaluate_candidate_sequences(candidate_action_sequences, obs)

            # pick the action sequence and return the 1st element of that sequence
            idx = np.argmax(predicted_rewards)
            best_action_sequence = candidate_action_sequences[idx]  # TODO (Q2)
            action_to_take = best_action_sequence[0]  # TODO (Q2)
            return action_to_take[None]  # Unsqueeze the first index

    def calculate_sum_of_rewards(self, obs, candidate_action_sequences, model):
        """

        :param obs: numpy array with the current observation. Shape [D_obs]
        :param candidate_action_sequences: numpy array with the candidate action
        sequences. Shape [N, H, D_action] where
            - N is the number of action sequences considered
            - H is the horizon
            - D_action is the action of the dimension
        :param model: The current dynamics model.
        :return: numpy array with the sum of rewards for each action sequence.
        The array should have shape [N].
        """

        N = candidate_action_sequences.shape[0]
        H = candidate_action_sequences.shape[1]
        # D_obs = obs.shape[0]

        obs_batch = np.tile(obs, (N,1))

        all_rewards = []
        for i in range(H):
            acs_batch = candidate_action_sequences[:, i, :]
            # if i == 0:
            #     print("acs_aaa", acs.shape)
            #     print("obs_aaa", obs_batch.shape)
            #     print("obs_aaa", obs_batch,shape)
            rewards, _ = self.env.get_reward(obs_batch, acs_batch)
            if i < H-1:
                obs_batch = model.get_prediction(obs_batch, acs_batch, self.data_statistics)
            # if i == 0:
            #     print("rewards_aaa", rewards.shape)
            all_rewards.append(rewards)
        sum_of_rewards = np.sum(all_rewards, axis = 0)

        # predicted_obs = []
        # for i in range(H):
        #     acs = candidate_action_sequences[:, i, :]
        #     obs_batch = model.get_prediction(obs_batch, acs, self.data_statistics)
        #     predicted_obs.append(obs_batch)
        # predicted_obs = np.array(predicted_obs)

        # sum_of_rewards = []
        # for j in range(N):
        #     acs = candidate_action_sequences[j, :, :]
        #     # print(predicted_obs.shape)
        #     obs_batch = predicted_obs[:, j, :]
        #     rewards, _ = self.env.get_reward(obs_batch, acs)
        #     sum_of_rewards.append(np.sum(rewards))
        # sum_of_rewards = np.array(sum_of_rewards)  

        # print("sum_of_rewards_after", sum_of_rewards.shape)
        # print("sum_of_rewards_after", sum_of_rewards,shape)

        
        # TODO (Q2)
        # For each candidate action sequence, predict a sequence of
        # states for each dynamics model in your ensemble.
        # Once you have a sequence of predicted states from each model in
        # your ensemble, calculate the sum of rewards for each sequence
        # using `self.env.get_reward(predicted_obs, action)`
        # You should sum across `self.horizon` time step.
        # Hint: you should use model.get_prediction and you shouldn't need
        #       to import pytorch in this file.
        # Hint: Remember that the model can process observations and actions
        #       in batch, which can be much faster than looping through each
        #       action sequence.
        return sum_of_rewards
