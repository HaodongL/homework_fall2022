import numpy as np


class ArgMaxPolicy(object):

    def __init__(self, critic):
        self.critic = critic

    def get_action(self, obs):
        if len(obs.shape) > 3:
            observation = obs
        else:
            observation = obs[None]
        
        ## TODO return the action that maxinmizes the Q-value 
        # at the current observation as the output
        qa_values = self.critic.qa_values(obs)
        action = np.argmax(qa_values)

        # print("shape a", action)
        # print("shape q", qa_values.shape)
        return action.squeeze()