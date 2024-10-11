import numpy as np
import socialforce
from crowd_sim.envs.policy.policy import Policy
from crowd_sim.envs.utils.action import ActionXY


class SocialForce(Policy):
    def __init__(self):
        super().__init__()
        self.name = 'socialforce'
        self.trainable = False
        self.multiagent_training = None
        self.kinematics = 'holonomic'
        self.initial_speed = 0.5
        self.v0 = 10
        self.sigma = 0.3
        self.sim = None
        self.time_step = 0.25

    def configure(self, config):
        return

    def set_phase(self, phase):
        return

    def predict(self, state, border=None):
        """

        :param state:
        :return:
        """
        sf_state = []
        self_state = state.robot_state
        sf_state.append((self_state.px, self_state.py, self_state.vx, self_state.vy, self_state.gx, self_state.gy))
        for human_state in state.human_states:
            # approximate desired direction with current velocity
            if human_state.vx == 0 and human_state.vy == 0:
                gx = np.random.random()
                gy = np.random.random()
            else:
                gx = human_state.px + human_state.vx
                gy = human_state.py + human_state.vy
            sf_state.append((human_state.px, human_state.py, human_state.vx, human_state.vy, gx, gy))
        sim = socialforce.Simulator(np.array(sf_state), delta_t=self.time_step, initial_speed=self.initial_speed,
                                    v0=self.v0, sigma=self.sigma)
        sim.step()
        action = ActionXY(sim.state[0, 2], sim.state[0, 3])
        #if border is not None and self.outside_check([sim.state[0, 0], sim.state[0, 1]], self_state.radius, border):
        #    action = ActionXY(0.0, 0.0)

        self.last_state = state

        return action


class CentralizedSocialForce(SocialForce):
    """
    Centralized socialforce, a bit different from decentralized socialforce, where the goal position of other agents is
    set to be (0, 0)
    """
    def __init__(self):
        super().__init__()
        self.time_step = 0.25

    def outside_check(self, position, radius, obstacle):
        left = position[0] - radius < obstacle[0][0]
        right = position[0] + radius > obstacle[1][0]
        below = position[1] - radius < obstacle[2][1]
        above = position[1] + radius > obstacle[1][1]
        if ((left or right) or (above or below)):
            return True

        return False

    def predict(self, state, border=None):
        sf_state = []
        for agent_state in state:
            sf_state.append((agent_state.px, agent_state.py, agent_state.vx, agent_state.vy,
                             agent_state.gx, agent_state.gy))

        sim = socialforce.Simulator(np.array(sf_state), delta_t=self.time_step, initial_speed=self.initial_speed,
                                    v0=self.v0, sigma=self.sigma)

        sim.step()
        actions = [ActionXY(sim.state[i, 2], sim.state[i, 3]) for i in range(len(state))]
        del sim

        return actions
