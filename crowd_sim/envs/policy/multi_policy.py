import numpy as np
import socialforce
from crowd_sim.envs.policy.policy import Policy
from crowd_sim.envs.utils.action import ActionXY
from crowd_sim.envs.policy.socialforce import CentralizedSocialForce, SocialForce
from crowd_sim.envs.policy.orca import CentralizedORCA, ORCA
from crowd_sim.envs.policy.linear import Linear

class CentralizedMultiPolicy(Policy):
    """
    Centralized planner for multiple agent policy types
    """
    def __init__(self):
        super().__init__()
        self.name = 'SocialForce'
        self.trainable = False
        self.multiagent_training = None
        self.kinematics = 'holonomic'
        self.initial_speed = 1
        self.v0 = 10
        self.sigma = 0.3
        self.sim = None

    def configure(self, config):
        return

    def set_phase(self, phase):
        return
    
    def predict(self, state, policies=None, orca_border=None, sfm_border=None):
        centralized_sf = CentralizedSocialForce()
        centralized_orca = CentralizedORCA()
        linear = Linear()

        sf_actions = centralized_sf.predict(state, sfm_border)

        orca_actions = centralized_orca.predict(state, orca_border)


        if policies is not None:
            actions = []
            for i in range(len(policies)):
                if isinstance(policies[i], SocialForce):
                    actions.append(sf_actions[i])
                elif isinstance(policies[i], ORCA):
                    actions.append(orca_actions[i])
                elif isinstance(policies[i], Linear):
                    actions.append(linear.predict(state[i]))

        return actions, [sf_actions[-1], orca_actions[-1]]